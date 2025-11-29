from typing import Generator
from transformer import Transformer
from token_embedder import TokenEmbedder

class LargeLanguageModel:
    emb_size: int
    num_heads: int
    tokenizer_model_name: str
    token_embedder: TokenEmbedder
    transformer_model: Transformer


    def __init__(self, tokenizer_model_name: str, emb_size: int, num_heads: int):
        self.tokenizer_model_name = tokenizer_model_name
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.loaded = False


    def load_model(self):
        # Simulate loading a large language model
        self.token_embedder = TokenEmbedder(model_name=self.tokenizer_model_name,emb_size=self.emb_size)
        print(f"Loading model: {self.tokenizer_model_name}")

        self.loaded = True


    def forward_pass(self, prompt: str)->Generator[str, None, None]:
        if not self.loaded:
            raise Exception("Model not loaded. Please load the model before generating text.")
        token_embeddings = self.token_embedder.convert_tokens_to_embedding(prompt, generate_positional_encodings=True)
        self.transformer_model = Transformer(token_embeddings=token_embeddings,
                                             token_embedder=self.token_embedder,
                                             emb_size=self.emb_size, num_heads=self.num_heads)
        # Simulate text generation
        for token in self.transformer_model.forward_pass():
            yield token



