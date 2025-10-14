from transformers import AutoTokenizer, AutoModel

model_name = "huawei-noah/TinyBERT_General_4L_312D"  # TinyBERT base variant

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Transformers are revolutionizing AI research."
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
