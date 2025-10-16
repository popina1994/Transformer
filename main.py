import torch
from transformer import Transformer

# README: https://jalammar.github.io/illustrated-transformer/
if __name__ == "__main__":
    # A 2×4 matrix
    X_in = torch.tensor([[1, 2, 3, 4],
                  [4, 5, 6, 7]], dtype=torch.float64)
    X_out = torch.tensor([[4, 5, 6, 7],
                  [1, 2, 3, 8]], dtype=torch.float64)
    num_heads = 2
    transformer = Transformer(X_in.shape[1], num_heads=num_heads)
    transformer.forward_pass(X_in=X_in, X_out=X_out)

