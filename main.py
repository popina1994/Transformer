import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def self_attention(X: torch.Tensor, num_heads: int) -> torch.Tensor:
    print(f"{X=}")
    # (emb_size, learning_dim)
    W_q = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64)
    # (emb_size, learning_dim)
    W_k = torch.tensor([[4, 5, 6], [1, 2, 3], [7, 8, 9]], dtype=torch.float64)
    # (emb_size, hidden_dim)
    W_v = torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=torch.float64)
    # Q represents what teh current query is looking for.
    # K represents "label".
    # V represents "actual information"
    # Dimensions of X: (X.rows, emb_size)

    Q = X @ W_q #(X.rows, learning_dim)
    K = X @ W_k # (X.rows, learning_dim)
    V = X @ W_v #(X.rows, hidden_dim)
    # Normalization: Q @ K / sqrt(d_k)
    print(f"{Q=} {Q.shape=}")
    print(f"{K=} {K.shape=}")
    print(f"{V=} {V.shape=}")

    # score represents to some extent some sort of correlation matrix.
    # The higher the entry, the higher the correlation between token_i and token_j
    # (X.rows, X.rows)
    score = Q @ K.T
    print(f"{score=} {score.shape=}")

    # h number of heads
    # d_k = d_v  = emb_size / h in multi-head
    # emb_size an embedding size
    emb_size = X.shape[1]
    d_k = emb_size / num_heads
    # Normalization to flatten the scores.
    score /= math.sqrt(d_k)
    # Softmax each row and get the activation.
    # row is 0th dimensions, column is 1st.
    # reducing over the i-th dimensions, while keeping all other indices fixed.
    soft_max_out = F.softmax(score, dim=1)
    print(f"{soft_max_out=} {soft_max_out.shape=}")
    # self attention
    # (X.num_rows, hidden_dim)
    Z = soft_max_out @ V
    print(f"{Z=} {Z.shape=}")
    # Self attention compresses each token and stores the interaction with other tokens
    # in the sentence.
    #
    return Z


def add_and_normalize(X_in: torch.Tensor, X_out: torch.Tensor) -> torch.Tensor:
    embedding_dim: int = int(X_in.shape[1])
    input_mat = X_in + X_out
    norm_add_X = torch.layer_norm(input = input_mat,
                                        normalized_shape=(embedding_dim,))

    return norm_add_X


def multi_head_self_attention(X_in: torch.Tensor, num_heads: int) -> torch.Tensor:
    heads_out = [self_attention(X=X_in, num_heads=num_heads) for _ in range(num_heads)]
    # X_in.num_rows, hidden_dim * num_heads
    head = torch.concat(heads_out, dim=1)
    print(f"{head=}")
    # (hidden_dim, emb_size)
    W_temp = torch.tensor([[3, 4, 7], [5, 6, 8], [7, 8, 9]], dtype=torch.float64)
    # (hidden_dim * num_heads, emb_size)
    W_0 = torch.concat([W_temp] * num_heads, dim=0)
    print(f"{head.shape=} {W_0.shape=}")

    # (X_in.num_rows, emb_size)
    # emb_size because of add + batch normalization
    multi_head_out = head @ W_0
    print(f"{multi_head_out=} {X_in=}")

    return multi_head_out

def encoder(X_in: torch.Tensor, num_heads: int) -> torch.Tensor:
    # TODO: text to positional embedding + embedding
    # TODO: encode decoder attention

    multi_head_out = multi_head_self_attention(X_in, num_heads)
    # (X_in.num_rows, emb_size)
    norm_multi_head = add_and_normalize(X_in, multi_head_out)

    emb_size = X_in.shape[1]
    linear_model = nn.Linear(emb_size, emb_size).double()
    linear_layer_out = linear_model(norm_multi_head)
    encoder_out = add_and_normalize(multi_head_out, linear_layer_out)

    return encoder_out


if __name__ == "__main__":
    # A 2×3 matrix
    X_in = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float64)
    num_heads = 3
    encoder_out = encoder(X_in, num_heads=num_heads)
    print(f"Encoder out{encoder_out=}")

