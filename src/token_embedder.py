
import torch
from transformers import BertTokenizerFast, AutoTokenizer

class TokenEmbedder:
    tokenizer: AutoTokenizer
    embeddings: torch.Tensor
    model_name: str
    emb_size: int
    def __init__(self, model_name: str, emb_size: int):
        self.model_name = model_name
        self.emb_size = emb_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self.emb_size, dtype=torch.float64)


    @property
    def vocab_size(self)-> int:
        return self.tokenizer.vocab_size

    def convert_token_id_to_embedding(self, token_id: torch.Tensor)-> tuple[torch.Tensor, str]:
        list_token: list[str] = self.tokenizer.convert_ids_to_tokens(token_id)
        token: str = list_token[0]
        token_ids_torch: torch.Tensor = torch.tensor(token_id, dtype=torch.long)
        token_embedding = self.embedding_layer(token_ids_torch)
        token_embedding[0] += self.positional_encodings[0]

        return token_embedding, token


    def convert_tokens_to_embedding(self, text: str, generate_positional_encodings: bool=False)-> torch.Tensor:
        tokens: list[str] = self.tokenizer.tokenize(text)
        if generate_positional_encodings:
            self.positional_encodings = self.positional_encoding(len(tokens), False)
        token_ids: list[int] = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids_torch: torch.Tensor = torch.tensor(token_ids, dtype=torch.long)
        token_embeddings = self.embedding_layer(token_ids_torch)
        for idx_token in range(len(tokens)):
            token_embeddings[idx_token] += self.positional_encodings[idx_token]

        return token_embeddings


    @staticmethod
    # Code from https://www.tensorflow.org/tutorials/text/transformer
    def get_angles(positions: torch.Tensor, indices: torch.Tensor, embed_size: int) -> torch.Tensor:
        exponents: torch.Tensor = (2 * (indices // 2)) / embed_size
        angle_rates: torch.Tensor = 1 / torch.pow(10000, exponents)
        return torch.ger(positions, angle_rates)


    def positional_encoding(self, num_tokens: int, batch_computation: bool) -> torch.Tensor:
        """
        Returns (num_tokens, emb_size) tensor where
        each row represent positional encoding of the corresponding position
        """
        positions = torch.arange(num_tokens)
        indices = torch.arange(self.emb_size)
        angle_rads = TokenEmbedder.get_angles(positions=positions,
                                              indices=indices, embed_size=self.emb_size)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads.unsqueeze(0) if batch_computation else angle_rads

        return pos_encoding