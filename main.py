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
    emb_size: int
    learning_dim: int
    hidden_dim: int
    num_heads: int

    @staticmethod
    def fill_out_matrix(num_rows: int, num_cols: int, shift: int) -> torch.Tensor:
        A = torch.tensor([[0] * num_cols] * num_rows, dtype=torch.float64)
        for idx in range(num_rows * num_cols):
            row_idx: int = idx // num_cols
            col_idx: int = idx % num_cols
            A[row_idx, col_idx] = idx + 1

        return A


    def __init__(self, emb_size: int, learning_dim: int, hidden_dim: int, num_heads: int) -> None:
        # (emb_size, learning_dim)
        self.W_q = self.fill_out_matrix(emb_size, learning_dim, 1)
        # (emb_size, learning_dim)
        self.W_k = self.fill_out_matrix(emb_size, learning_dim, 2)
        # (emb_size, hidden_dim)
        self.W_v = self.fill_out_matrix(emb_size, hidden_dim, 3)

        self.emb_size = emb_size
        self.learning_dim = learning_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads


    def forward_pass(self, X_in: torch.Tensor) -> torch.Tensor:
        # Dimensions of X_in: (X_in.rows, emb_size)
        print(f"{X_in=}")

        # Q represents what teh current query is looking for.
        # K represents "label".
        # V represents "actual information"
        Q = X_in @ self.W_q #(X_in.rows, learning_dim)
        K = X_in @ self.W_k #(X_in.rows, learning_dim)
        V = X_in @ self.W_v #(X_in.rows, hidden_dim)

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
        d_k = emb_size / num_heads
        # Normalization to flatten the scores.
        score /= math.sqrt(d_k)
        # Softmax each row and get the activation.
        # row is 0th dimensions, column is 1st.
        # reducing over the i-th dimensions, while keeping all other indices fixed.
        soft_max_out = F.softmax(score, dim=1)
        print(f"{soft_max_out=} {soft_max_out.shape=}")
        # self attention
        # (X_in.num_rows, hidden_dim)
        Z = soft_max_out @ V
        print(f"{Z=} {Z.shape=}")
        # Self attention compresses each token and stores the interaction with other tokens
        # in the sentence.
        return Z

@dataclass
class MultiHeadSelfAttention:
    emb_size: int
    learning_dim: int
    hidden_dim: int
    num_heads: int
    W_multi_head: torch.Tensor
    self_attention_models: list[SelfAttentionModel]

    def __init__(self, emb_size: int, learning_dim: int, hidden_dim: int, num_heads: int):
        self.emb_size = emb_size
        self.learning_dim = learning_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.self_attention_models = [SelfAttentionModel(emb_size=emb_size, learning_dim=learning_dim,
                                              hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_heads)]

    def forward_pass(self, X_in: torch.Tensor) -> torch.Tensor:
        heads_out = [self_attention_model.forward_pass(X_in) for self_attention_model in self.self_attention_models]
        # X_in.num_rows, hidden_dim * num_heads
        head = torch.concat(heads_out, dim=1)
        print(f"{head=}")
        # (hidden_dim, emb_size)
        W_temp = SelfAttentionModel.fill_out_matrix(self.hidden_dim, self.emb_size, 3)
        # (hidden_dim * num_heads, emb_size)
        self.W_multi_head = torch.concat([W_temp] * num_heads, dim=0)
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

    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.multi_head_self_attention = MultiHeadSelfAttention(emb_size=emb_size,
                                        learning_dim=emb_size + 1,
                                        hidden_dim=emb_size + 2, num_heads=num_heads)


    def forward_pass(self, X_in: torch.Tensor) -> torch.Tensor:
        multi_head_out = self.multi_head_self_attention.forward_pass(X_in)
        # (X_in.num_rows, emb_size)
        norm_multi_head = add_and_normalize(X_in, multi_head_out)

        emb_size = X_in.shape[1]
        linear_model = nn.Linear(emb_size, emb_size).double()
        linear_layer_out = linear_model(norm_multi_head)
        encoder_out = add_and_normalize(multi_head_out, linear_layer_out)

        return encoder_out


def masked_multi_head_attention(X_in: torch.Tensor, num_heads: int):
    # TODO: masking
    # setting entries to infinity in the attention matrices so the future tokens do not affect
    return None #multi_head_self_attention(X_in, num_heads)


def decoder(X_in: torch.Tensor, num_heads: int)-> torch.Tensor:
    masked_multi_head_out = masked_multi_head_attention(X_in, num_heads)
    norm_multi_head = add_and_normalize(X_in, masked_multi_head_out)
    # decoder_encoder_attention = decoder_encoder_attention(X_in, num_heads)

    # K, V from encoder, Q from decoder (makes sense, we are querying)
    #  -> encoder-decoder attention, add & norm

# def post_encoding
# feedforward
# add & norm

if __name__ == "__main__":
    # A 2×3 matrix
    X_in = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float64)
    num_heads = 3
    encoder = Encoder(emb_size=X_in.shape[1], num_heads=num_heads)
    encoder_out = encoder.forward_pass(X_in=X_in)
    print(f"Encoder out{encoder_out=}")

