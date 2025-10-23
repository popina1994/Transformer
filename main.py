import torch
from transformer import Transformer

# README: https://jalammar.github.io/illustrated-transformer/
if __name__ == "__main__":
    # A 2×4 matrix
    X_out = torch.tensor([[4, 5, 6, 7],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],], dtype=torch.float64)
    num_heads = 2
    emb_size = 4
    text = "Transformers are revolutionizing AI research."
    transformer = Transformer(emb_size=emb_size, num_heads=num_heads, text=text)
    for token in transformer.forward_pass(X_out=X_out):
        print(token)

    print("SUCCESS")