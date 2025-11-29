from llm import LargeLanguageModel

# README: https://jalammar.github.io/illustrated-transformer/
if __name__ == "__main__":
    num_heads = 2
    emb_size = 16
    text = "Transformers are revolutionizing AI research."
    llm = LargeLanguageModel(tokenizer_model_name="bert-base-uncased", emb_size=emb_size, num_heads=num_heads)
    llm.load_model()
    for token in llm.forward_pass(text):
        print(f"Next token is: {token}", end=" ")

    print(" Text generation completed. ")