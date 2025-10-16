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
