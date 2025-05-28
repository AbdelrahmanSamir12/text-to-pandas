import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, TrainingArguments , Trainer
)
import evaluate
# Set parameters
MAX_INPUT = 128  
MAX_OUTPUT = 128



def main():
    # Load data
    train_df = pd.read_csv("data/processed/train_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    # Convert to Hugging Face Dataset
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(test_df)

    # Load model and tokenizer
    checkpoint = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Preprocess function
    def preprocess(batch):
        inputs = tokenizer(batch["Input"], truncation=True, padding="max_length", max_length=MAX_INPUT)
        outputs = tokenizer(batch["Pandas Query"], truncation=True, padding="max_length", max_length=MAX_OUTPUT)
        inputs["labels"] = outputs["input_ids"]
        return inputs

    # Tokenize datasets
    train_ds = train_ds.map(preprocess, batched=True)
    eval_ds = eval_ds.map(preprocess, batched=True)

    # Model
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/text-to-pandas-bart",
        #evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=2,
        #predict_with_generate=True,
        logging_dir='./logs',
        logging_steps=15,
        fp16=True, 
        #save_strategy="epoch",
        #load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save model
    model_dir = "models/text-to-pandas-bart4"
    trainer.save_model(model_dir)

    # save tokenizer
    tokenizer.save_pretrained(f"{model_dir}")

    print("Model and Tokenizer saved successfully.")

if __name__ == '__main__':
    main()