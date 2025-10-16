from dataclasses import dataclass
from encoder import Encoder
from decoder import Decoder
import torch

@dataclass
class Transformer:
    emb_size: int
    num_heads: int
    encoder: Encoder
    decoder: Decoder

    def __init__(self, emb_size: int, num_heads: int):
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.encoder = Encoder(emb_size=emb_size, num_heads=num_heads)
        self.decoder = Decoder(emb_size=emb_size, num_heads=num_heads)


    def forward_pass(self, X_in: torch.Tensor, X_out: torch.Tensor)->torch.Tensor:
        encoder_output = self.encoder.forward_pass(X_in=X_in)
        print(f"{encoder_output=}")
        decoder_out = self.decoder.forward_pass(X_in=X_out, encoder_output=encoder_output)
        print(f"Decoder out{decoder_out=}")

        # TODO loop to feed the decoder the number of times we have the input.
        return decoder_out