from typing import Counter
from datasets import Value, concatenate_datasets, load_dataset

combined = load_dataset("csv", data_files="combined_dataset.csv")['train']

label_counts = Counter(combined["label"])
print(label_counts)