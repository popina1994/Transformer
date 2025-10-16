import torch
from encoder import Encoder
from decoder import Decoder

def masked_multi_head_attention(X_in: torch.Tensor, num_heads: int):
    # TODO: masking
    # setting entries to infinity in the attention matrices so the future tokens do not affect
    return None #multi_head_self_attention(X_in, num_heads)


# def post_encoding
# feedforward
# add & norm

if __name__ == "__main__":
    # A 2×3 matrix
    X_in = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float64)
    X_out = torch.tensor([[4, 5, 6],
                  [1, 2, 3]], dtype=torch.float64)
    num_heads = 3
    encoder = Encoder(emb_size=X_in.shape[1], num_heads=num_heads)
    encoder_out = encoder.forward_pass(X_in=X_in)
    print(f"Encoder out{encoder_out=}")
    decoder = Decoder(emb_size=X_in.shape[1], num_heads=num_heads)
    decoder_out = decoder.forward_pass(X_in=X_out, encoder_K=encoder.get_K(), encoder_V=encoder.get_V())
    print(f"Decoder out{decoder_out=}")

