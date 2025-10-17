from dataclasses import dataclass
from encoder import Encoder
from decoder import Decoder
import torch
import torch.nn.functional as F
from typing import Generator

@dataclass
class Transformer:
    emb_size: int
    num_heads: int
    vocab: list[str]

    encoder: Encoder
    decoder: Decoder

    def __init__(self, emb_size: int, num_heads: int, vocab: list[str]):
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.encoder = Encoder(emb_size=emb_size, num_heads=num_heads)
        self.decoder = Decoder(emb_size=emb_size, num_heads=num_heads)
        self.linear_layer = torch.nn.Linear(emb_size, len(vocab)).double()
        self.vocab = vocab


    @staticmethod
    # Code from https://www.tensorflow.org/tutorials/text/transformer
    def get_angles(positions: torch.Tensor, indices: torch.Tensor, embed_size: int) -> torch.Tensor:
        exponents: torch.Tensor = (2 * (indices // 2)) / embed_size
        angle_rates: torch.Tensor = 1 / torch.pow(10000, exponents)
        return positions * angle_rates


    def positional_encoding(self, num_tokens: int, batch_computation: bool) -> torch.Tensor:
        """
        Returns (num_tokens, emb_size) tensor where
        each row represent positional encoding of the corresponding position
        """
        positions = torch.arange(num_tokens).unsqueeze(1)
        indices = torch.arange(self.emb_size).unsqueeze(1)
        angle_rads = Transformer.get_angles(positions=positions,
                                            indices=indices, embed_size=self.emb_size)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads.unsqueeze(0) if batch_computation else angle_rads

        return pos_encoding


    def forward_pass(self, X_in: torch.Tensor, X_out: torch.Tensor)->Generator[str, None, None]:
        encoder_output = self.encoder.forward_pass(X_in=X_in)
        print(f"{encoder_output=}")
        num_tokens = X_out.shape[0]
        for i in range(num_tokens - 1):
            decoder_out = self.decoder.forward_pass(X_in=X_out[0:(i+2), :], encoder_output=encoder_output)
            print(f"{decoder_out=}")

            # Get the logits for the currently encoded sequence of words + the input query.
            logits = self.linear_layer(decoder_out)[i+1]
            # Check which of the words is to be the next one using probabilities with
            # softmax
            logits_prob = F.softmax(logits)
            print(f"{logits_prob=}")
            max_idx_word = torch.argmax(logits_prob)
            # get the word from the vocabulary
            word = self.vocab[max_idx_word]
            yield word
            # encode word and position
            #X_out[i+1, :] = map_word_to_tensor(word, i+1)
