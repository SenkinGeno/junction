from typing import Counter
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

ds = load_dataset("csv", data_files="generated_extremist.csv")['train']

label_counts = Counter(ds["label"])
print(label_counts)
