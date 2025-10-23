import torch
import torch.nn as nn
from dataclasses import dataclass
from encoder import add_and_normalize
from encoder import MultiHeadSelfAttention
from typing import override

@dataclass
class MultiHeadMaskedSelfAttention(MultiHeadSelfAttention):
    def __init__(self, emb_size: int,  num_heads: int,
                 pass_K_and_V: bool = False):
        super().__init__(emb_size=emb_size,
                          num_heads=num_heads)

    #TODO: add forward pass with the masking logic.

@dataclass
class EncoderDecoderMultiHeadSelfAttention(MultiHeadSelfAttention):
    def __init__(self, emb_size: int, num_heads: int):
        super().__init__(emb_size=emb_size, num_heads=num_heads)

    @override
    def forward_pass(self, X_in: torch.Tensor,
                     encoder_output: torch.Tensor) -> torch.Tensor:
        return super().forward_pass(X_in, encoder_output=encoder_output)



@dataclass
class Decoder:
    emb_size: int
    num_heads: int
    multi_head_self_attention: MultiHeadMaskedSelfAttention
    encoder_decoder_attention: EncoderDecoderMultiHeadSelfAttention
    linear_mode: nn.Linear

    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.multi_head_self_attention = MultiHeadMaskedSelfAttention(emb_size=emb_size,
                                        num_heads=num_heads)
        self.encoder_decoder_attention = EncoderDecoderMultiHeadSelfAttention(emb_size=emb_size, num_heads=num_heads)
        self.linear_model = nn.Linear(emb_size, emb_size).double()


    def forward_pass(self, X_in: torch.Tensor,
                    encoder_output: torch.Tensor):
        #TODO: masked self attention
        print(f"X_input: {X_in=}")
        multi_head_out = self.multi_head_self_attention.forward_pass(X_in=X_in)
        # (X_in.num_rows, emb_size)
        norm_multi_head = add_and_normalize(X_in, multi_head_out)

        # encoder decoder attention
        encoder_decoder_attention = \
            self.encoder_decoder_attention.forward_pass(norm_multi_head, encoder_output=encoder_output)
        norm_encoder_decoder_attention = add_and_normalize(X_in, encoder_decoder_attention)

        # feedforward
        linear_layer_out = self.linear_model(norm_encoder_decoder_attention)
        # add & norm
        decoder_out = add_and_normalize(multi_head_out, linear_layer_out)

        return decoder_out


