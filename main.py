import torch
from transformer import Transformer

# README: https://jalammar.github.io/illustrated-transformer/
if __name__ == "__main__":
    # A 2×4 matrix
    X_in = torch.tensor([[1, 2, 3, 4],
                  [4, 5, 6, 7],
                  [8, 9, 10, 11]], dtype=torch.float64)
    X_out = torch.tensor([[4, 5, 6, 7],
                  [1, 2, 3, 8],
                  [20, 21, 22, 23]], dtype=torch.float64)
    # TODO: encoding of words as tensors + position
    num_heads = 2
    vocab = ["Mama", "Tata"]
    transformer = Transformer(X_in.shape[1], num_heads=num_heads, vocab=vocab)
    for token in transformer.forward_pass(X_in=X_in, X_out=X_out):
        print(token)

