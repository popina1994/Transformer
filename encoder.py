import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def add_and_normalize(X_in: torch.Tensor, X_out: torch.Tensor) -> torch.Tensor:
    embedding_dim: int = int(X_in.shape[1])
    input_mat = X_in + X_out
    norm_add_X = torch.layer_norm(input = input_mat,
                                        normalized_shape=(embedding_dim,))

    return norm_add_X

@dataclass
class SelfAttentionModel:
    W_q: torch.Tensor
    W_k: torch.Tensor
    W_v: torch.Tensor
    emb_size: int = 1
    d_k: int = 1
    d_v: int = 1
    num_heads: int = 1

    @staticmethod
    def fill_out_matrix(num_rows: int, num_cols: int, shift: int) -> torch.Tensor:
        A = torch.tensor([[0] * num_cols] * num_rows, dtype=torch.float64)
        for idx in range(num_rows * num_cols):
            row_idx: int = idx // num_cols
            col_idx: int = idx % num_cols
            A[row_idx, col_idx] = idx + 1

        return A


    def __init__(self, emb_size: int, d_k: int, d_v: int, num_heads: int) -> None:
        # (emb_size, d_k)
        self.W_q = self.fill_out_matrix(emb_size, d_k, 1)

        # (emb_size, d_k)
        self.W_k = self.fill_out_matrix(emb_size, d_k, 2)
        # (emb_size, d_v)
        self.W_v = self.fill_out_matrix(emb_size, d_v, 3)

        self.emb_size = emb_size
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads


    def forward_pass(self, X_in: torch.Tensor,
                     encoder_output: torch.Tensor | None = None) -> torch.Tensor:
        # Dimensions of X_in: (X_in.rows, emb_size)
        print(f"{X_in=}")

        # Q represents what teh current query is looking for.
        # K represents "label".
        # V represents "actual information"
        Q = X_in @ self.W_q #(X_in.rows, d_k)

        X_kin = X_in if encoder_output is None else encoder_output
        X_vin = X_in if encoder_output is None else encoder_output
        K: torch.Tensor = X_kin @ self.W_k  #(X_in.rows, d_k)
        V: torch.Tensor = X_vin @ self.W_v  #(X_in.rows, d_v)
        self.mat_K = K
        self.mat_V = V

        print(f"{Q=} {Q.shape=}")
        print(f"{K=} {K.shape=}")
        print(f"{V=} {V.shape=}")

        # score represents to some extent some sort of correlation matrix.
        # The higher the entry, the higher the correlation between token_i and token_j
        # (X_in.rows, X_in.rows)
        score = Q @ K.T
        print(f"{score=} {score.shape=}")

        # h number of heads
        # d_k = d_v  = emb_size / h in multi-head
        # emb_size an embedding size
        emb_size = X_in.shape[1]
        d_k = emb_size / self.num_heads
        # Normalization to flatten the scores.
        score /= math.sqrt(d_k)
        # Softmax each row and get the activation.
        # row is 0th dimensions, column is 1st.
        # applying over the i-th dimensions, while keeping all other indices fixed.
        soft_max_out = F.softmax(score, dim=1)
        print(f"{soft_max_out=} {soft_max_out.shape=}")
        # self attention
        # (X_in.num_rows, d_v)
        Z = soft_max_out @ V
        print(f"{Z=} {Z.shape=}")
        # Self attention compresses each token and stores the interaction with other tokens
        # in the sentence.
        return Z

@dataclass
class MultiHeadSelfAttention:
    emb_size: int
    d_k: int
    d_v: int
    num_heads: int
    W_multi_head: torch.Tensor
    self_attention_models: list[SelfAttentionModel]
    mat_K: list[torch.Tensor]
    mat_V: list[torch.Tensor]

    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.d_k = emb_size // num_heads
        self.d_v = emb_size // num_heads
        self.num_heads = num_heads
        self.self_attention_models = [SelfAttentionModel(emb_size=emb_size,
                                                        d_k=self.d_k,
                                                        d_v=self.d_v,
                                                        num_heads=num_heads)
                                                        for _ in range(num_heads)]

    def forward_pass(self, X_in: torch.Tensor,
                     encoder_output: torch.Tensor | None = None) -> torch.Tensor:
        heads_out = [self_attention_model.forward_pass(X_in, encoder_output) for self_attention_model in self.self_attention_models]

        # X_in.num_rows, d_v * num_heads
        head = torch.concat(heads_out, dim=1)
        print(f"{head=}")
        # (d_v, emb_size)
        W_temp = SelfAttentionModel.fill_out_matrix(self.d_v, self.emb_size, 3)
        # (d_v * num_heads, emb_size)
        self.W_multi_head = torch.concat([W_temp] * self.num_heads, dim=0)
        print(f"{head.shape=} {self.W_multi_head.shape=}")

        # (X_in.num_rows, emb_size)
        # emb_size because of add + batch normalization
        multi_head_out = head @ self.W_multi_head
        print(f"{multi_head_out=} {X_in=}")

        return multi_head_out

@dataclass
class Encoder:
    emb_size: int
    num_heads: int
    multi_head_self_attention: MultiHeadSelfAttention
    linear_mode: nn.Linear

    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.multi_head_self_attention = MultiHeadSelfAttention(emb_size=emb_size,
                                                                num_heads=num_heads)
        self.linear_model = nn.Linear(emb_size, emb_size).double()


    def forward_pass(self, X_in: torch.Tensor) -> torch.Tensor:
        multi_head_out = self.multi_head_self_attention.forward_pass(X_in)
        # (X_in.num_rows, emb_size)
        norm_multi_head = add_and_normalize(X_in, multi_head_out)

        linear_layer_out = self.linear_model(norm_multi_head)
        encoder_out = add_and_normalize(multi_head_out, linear_layer_out)

        return encoder_out
