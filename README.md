# text-to-pandas

fine-tuned BART model to generate executable pandas code from natural language questions on tabular data.

---

## 🚀 Overview

**text-to-pandas** converts schema-rich natural language prompts into executable pandas expressions. It uses a BART sequence-to-sequence model, fine-tuned on a dataset of table schemas and associated queries.

---

## 📂 Project Structure
```
.
├── data
│ ├── external/
│ ├── interim/
│ ├── processed/
│ │ ├── test_clean.csv
│ │ └── train_clean.csv
│ └── raw/
│ ├── test.csv
│ └── train.csv
├── docs/
├── models/
│ └── text-to-pandas-bart/
├── notebooks/
│ └── data_exploration.ipynb
├── pyproject.toml
├── references/
├── reports/
│ └── figures/
├── results/
│ └── eval_predictions.csv
├── src/
│ ├── data_downloader.py
│ ├── evaluate.py
│ ├── preprocess.py
│ ├── test_model.py
│ └── train.py
└── uv.lock
```

---

## 📝 Data

Data is sourced from [Hugging Face: Rahima411/text-to-pandas](https://huggingface.co/datasets/Rahima411/text-to-pandas) and preprocessed.

- Raw: `data/raw/`
- Cleaned: `data/processed/`

Each entry gives a table schema, a natural language question, and the correct pandas query.

---

## 🛠️ Tech Stack

- **Python** – Core language and scripting
- **uv** – Super-fast Python package & workflow manager [astral-sh/uv](https://github.com/astral-sh/uv)
- **Hugging Face 🤗 Transformers** – For BART model and tokenization
- **PyTorch** – Deep learning backend for model training/inference
- **pandas** – Data manipulation and evaluation
- **Hugging Face Datasets** – Data loading and preprocessing
- **Jupyter Notebooks** – For data exploration and quick prototyping
- **Structured project layout** inspired by Cookiecutter Data Science
---

## 📚 License
MIT License