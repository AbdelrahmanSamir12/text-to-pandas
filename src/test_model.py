from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# Replace with the correct path if needed
model_dir = "models/text-to-pandas-bart2"  #
tokeinzer_dir = "models/text-to-pandas-bart_tokenizer2" 

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

def generate_pandas_code(inputs, max_input_len=128, max_gen_len=128):
    # Accept either a list or a single string
    single = isinstance(inputs, str)
    if single:
        inputs = [inputs]
    encodings = tokenizer(inputs, truncation=True, padding=True, max_length=max_input_len, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**encodings, max_new_tokens=max_gen_len)
    preds = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
    return preds[0] if single else preds

# Example usage
input_text = """table: table_name_9 (races : int64 , series : object , podiums : object)
question: What is the sum of Races, when Series is Toyota Racing Series, and when Podiums is greater than 3?"""
print(generate_pandas_code(input_text))