import torch
from encoder import Encoder

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

