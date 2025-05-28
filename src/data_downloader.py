
from datasets import load_dataset

# This will download and load the dataset; you can specify a split ("train", "test", etc.)
dataset = load_dataset("Rahima411/text-to-pandas", split="train")
test = load_dataset("Rahima411/text-to-pandas", split="test")
# Convert to pandas DataFrame
df = dataset.to_pandas()

# Save to local CSV
df.to_csv("data/raw/train.csv", index=False)

# save the test set to a CSV file
test_df = test.to_pandas()
test_df.to_csv("data/raw/test.csv", index=False)