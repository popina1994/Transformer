import torch
from transformer import Transformer

# README: https://jalammar.github.io/illustrated-transformer/
if __name__ == "__main__":
    num_heads = 2
    emb_size = 4
    text = "Transformers are revolutionizing AI research."
    transformer = Transformer(emb_size=emb_size, num_heads=num_heads, text=text)
    for token in transformer.forward_pass():
        print(token)

    print("SUCCESS")