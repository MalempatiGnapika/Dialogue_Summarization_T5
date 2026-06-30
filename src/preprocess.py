import pandas as pd
import re

# Load dataset
train_data = pd.read_csv("data/samsum-train.csv")
val_data = pd.read_csv("data/samsum-validation.csv")

# Optional: Use a smaller subset for faster training
train_data = train_data.sample(n=4000, random_state=42).reset_index(drop=True)
val_data = val_data.sample(n=500, random_state=42).reset_index(drop=True)

# Clean text
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

# Apply cleaning
train_data["dialogue"] = train_data["dialogue"].apply(clean_data)
train_data["summary"] = train_data["summary"].apply(clean_data)

val_data["dialogue"] = val_data["dialogue"].apply(clean_data)
val_data["summary"] = val_data["summary"].apply(clean_data)

print("Training Samples:", len(train_data))
print("Validation Samples:", len(val_data))

print(train_data.head())