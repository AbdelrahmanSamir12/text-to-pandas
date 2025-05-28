from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from tqdm import tqdm
import re
import os

def exact_match(pred, gold):
    return pred.strip().lower() == gold.strip().lower()

def normalized_match(pred, gold):
    """More lenient matching that handles formatting differences"""
    # Remove extra whitespace and normalize
    pred_norm = re.sub(r'\s+', ' ', pred.strip().lower())
    gold_norm = re.sub(r'\s+', ' ', gold.strip().lower())
    #remove all white spaces
    pred_norm = re.sub(r'\s+', '', pred_norm)
    gold_norm = re.sub(r'\s+', '', gold_norm)
    
    return pred_norm == gold_norm

def evaluate():
    # Load the model and tokenizer
    model_dir = "models/text-to-pandas-bart"  

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    model.eval()

    test = pd.read_csv("data/processed/test_clean.csv")

    # Number of samples to evaluate
    num_samples = 100
    correct_exact = 0
    correct_normalized = 0

    # Lists to store results for CSV
    inputs = []
    ground_truths = []
    predictions = []
    exact_matches = []

    for i in tqdm(range(num_samples)):
        input_text = test['Input'].iloc[i]
        true_pandas = test['Pandas Query'].iloc[i]

        tokenizer_inputs = tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **tokenizer_inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        pred_pandas = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check both exact and normalized matches
        is_exact_match = exact_match(pred_pandas, true_pandas)
        if is_exact_match:
            correct_exact += 1
            correct_normalized += 1
        elif normalized_match(pred_pandas, true_pandas):
            correct_normalized += 1

        # Store results for CSV
        inputs.append(input_text)
        ground_truths.append(true_pandas)
        predictions.append(pred_pandas)
        exact_matches.append(is_exact_match)

        print(f"Generated Query: {pred_pandas}")
        print(f"True Query: {true_pandas}")
        print(f"Exact Match: {is_exact_match}")
        print(f"Normalized Match: {normalized_match(pred_pandas, true_pandas)}")
        print("-" * 50)
    
    # Fixed: divide by actual number of samples evaluated, not total dataset size
    exact_accuracy = correct_exact / num_samples
    normalized_accuracy = correct_normalized / num_samples
    
    print(f"Exact Match Accuracy: {exact_accuracy:.2%}")
    print(f"Normalized Match Accuracy: {normalized_accuracy:.2%}")
    print(f"Evaluated {num_samples} samples out of {len(test)} total")

    # Create results DataFrame and save to CSV
    results_df = pd.DataFrame({
        "Input": inputs,
        "Ground_Truth": ground_truths,
        "Prediction": predictions,
        "Exact_Match": exact_matches
    })
    
    
    # Save to CSV
    results_df.to_csv("results/eval_predictions.csv", index=False)
    print(f"Results saved to results/eval_predictions.csv")

if __name__ == '__main__':
    evaluate()