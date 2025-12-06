from dataclasses import dataclass
from encoder import Encoder
from decoder import Decoder
import torch
import torch.nn.functional as F
from typing import Generator
from token_embedder import TokenEmbedder

@dataclass
class Transformer:
    encoder: Encoder
    decoder: Decoder
    token_embeddings: torch.Tensor
    encoder_output: torch.Tensor
    token_embedder: TokenEmbedder
    emb_size: int
    use_kv_cache: bool = False

    def __init__(self, num_heads: int, emb_size: int, token_embeddings: torch.Tensor,
                 token_embedder: TokenEmbedder=None, use_kv_cache: bool = False):
        torch.manual_seed(seed=42)
        self.emb_size = emb_size
        num_heads = num_heads
        self.use_kv_cache = use_kv_cache
        self.encoder = Encoder(emb_size=self.emb_size, num_heads=num_heads)
        self.decoder = Decoder(emb_size=self.emb_size, num_heads=num_heads, use_kv_cache=use_kv_cache)
        self.token_embeddings = token_embeddings
        self.token_embedder = token_embedder
        self.decoder_out_linear_layer = torch.nn.Linear(in_features=emb_size, out_features=self.token_embedder.vocab_size, ).double()
        self.encoder_output = self.encoder.forward_pass(X_in=self.token_embeddings)
        print(f"{self.token_embeddings=}")
        print(f"{self.encoder_output=}")


    def forward_pass(self)->Generator[str, None, None]:
        num_tokens = self.token_embeddings.shape[0]
        list_idxs: list[float] = [float(elem) for elem in torch.arange(self.emb_size)]
        initial_tokens = [[1] * self.emb_size for _ in range(num_tokens - 1)]
        initial_tokens = [list_idxs] + initial_tokens
        X_out = torch.tensor(initial_tokens, dtype=torch.float64)
        print(f"{X_out.shape=} {X_out=}")
        for i in range(num_tokens - 1):
            decoder_out = self.decoder.forward_pass(X_in=X_out[0:(i+2), :],
                                                    encoder_output=self.encoder_output)
            print(f"{decoder_out=} {X_out=}")

            # Get the logits for the currently encoded sequence of words + the input query.
            logits = self.decoder_out_linear_layer(decoder_out)[i+1]
            # Check which of the words is to be the next one using probabilities with
            # softmax
            logits_prob = F.softmax(logits)
            print(f"{logits_prob=}")
            token_idx = torch.argmax(logits_prob)
            # get the word from the vocabulary
            token_idx_tensor = torch.tensor([token_idx], dtype=torch.float64)
            token_embedding, token = self.token_embedder.convert_token_id_to_embedding(token_idx_tensor)
            print(f"{X_out=} {token_embedding=}")
            X_out[i+1, :] = token_embedding
            print(f"{X_out=}")
            yield token
            # encode word and position
