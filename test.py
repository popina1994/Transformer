from transformers import AutoTokenizer, AutoModel

model_name = "huawei-noah/TinyBERT_General_4L_312D"  # TinyBERT base variant

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "Transformers are revolutionizing AI research."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
# model = AutoModel.from_pretrained(model_name)
print(tokens)
print(token_ids)
# inputs = tokenizer(text, return_tensors="pt")
# print(inputs["input_ids"])
# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)
