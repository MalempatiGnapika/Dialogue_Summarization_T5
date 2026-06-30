import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)

# Load cleaned data
train_df = pd.read_csv("data/samsum-train.csv")
val_df = pd.read_csv("data/samsum-validation.csv")

# Smaller subset (remove these lines later if you want full training)
train_df = train_df.sample(n=4000, random_state=42)
val_df = val_df.sample(n=500, random_state=42)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenization
def preprocess(example):
    inputs = tokenizer(
        example["dialogue"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["summary"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess, batched=False)
val_dataset = val_dataset.map(preprocess, batched=False)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Training settings
training_args = TrainingArguments(
    output_dir="./saved_summary_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    save_strategy="epoch",
    logging_steps=100
    # ,evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
    # ,eval_dataset=val_dataset
)

trainer.train()

model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")

print("Training Complete!")