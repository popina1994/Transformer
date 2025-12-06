from llm import LargeLanguageModel
import argparse

# README: https://jalammar.github.io/illustrated-transformer/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the transformer LLM")
    parser.add_argument("--num-heads", type=int, default=2, help="number of attention heads (default: 2)")
    parser.add_argument("--emb-size", type=int, default=16, help="embedding size (default: 16)")
    parser.add_argument("--use-kv-cache", action="store_true", default=False, help="enable key/value cache (default: False)")
    parser.add_argument("text", help="input text for generation")

    args = parser.parse_args()
    num_heads = args.num_heads
    emb_size = args.emb_size
    use_kv_cache = args.use_kv_cache
    text = args.text

    llm = LargeLanguageModel(tokenizer_model_name="bert-base-uncased", emb_size=emb_size, num_heads=num_heads, use_kv_cache=use_kv_cache)
    llm.load_model()
    for token in llm.forward_pass(text):
        print(f"Next token is: {token}", end=" ")

    print(" Text generation completed. ")