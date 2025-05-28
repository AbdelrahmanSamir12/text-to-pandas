import pandas as pd
import re

def clean_schema(text):
    """Convert schema format from 'Table Name: table (field (type))' to 'table: table (field : type)'"""
    
    # 1. Normalize the table name prefix
    text = text.strip().replace("Table Name:", "table:")
    
    # 2. Replace all (type) with : type
    text = re.sub(r'(\w+)\s*\(\s*(\w+)\s*\)', r'\1 : \2', text)
    
    # 3. Fix spacing around commas
    text = re.sub(r'\s*,\s*', ' , ', text)
    
    return text

    # def clean_schema(text):
    #     # "Table Name: head (age (int64))" => "table: head (age: int64)"
    #     text = text.strip()
    #     text = text.replace("Table Name:", "table:") # normalize tables
    #     text = re.sub(r'(\w+) $$(\w+)$$', r'\1: \2', text)
    #     text = re.sub(' +', ' ', text)
    # return text

    
def clean_question(text):
    text = text.strip()
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    if not text.lower().startswith('question:'):
        text = f"question: {text}"
    return text

def clean_input(text):
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    schema_lines = [l for l in lines if l.lower().startswith("table name:")]
    other_lines = [l for l in lines if not l.lower().startswith("table name:")]
    
    # Clean each detected schema line
    cleaned_schemas = [clean_schema(schema) for schema in schema_lines]

    schemas_part = '\n'.join(cleaned_schemas)
    
    # the rest is the question 
    question_part = clean_question(' '.join(other_lines))
    
    # Combine schemas and question, separate with double newline for clarity
    if schemas_part:
        return f"{schemas_part}\n{question_part}"
    else:
        return question_part

def clean_output(code):
    code = code.strip()
    code = code.replace("’", "'").replace('“', '"').replace('”', '"')
    return code

def prepare_for_bart(infile, outfile, input_col="Input", output_col="Pandas Query"):
    df = pd.read_csv(infile)
    df = df[[input_col, output_col]].dropna()
    df[input_col] = df[input_col].apply(clean_input)
    df[output_col] = df[output_col].apply(clean_output)
    df.to_csv(outfile, index=False)

if __name__ == '__main__':
    prepare_for_bart('data/raw/train.csv', 'data/processed/train_clean.csv')
    prepare_for_bart('data/raw/test.csv', 'data/processed/test_clean.csv')