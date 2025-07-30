# train_sql_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Load custom data
df = pd.read_csv("data.csv")
df = df.dropna()
train_df, val_df = train_test_split(df, test_size=0.2)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model (no sentencepiece)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Tokenize
def tokenize(batch):
    inputs = tokenizer(batch["question"], padding="max_length", truncation=True, max_length=64)
    targets = tokenizer(batch["query"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Training setup
args = TrainingArguments(
    output_dir="sql_bart_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    logging_dir="./logs",
    save_total_limit=1,
    logging_steps=5,
    do_eval=False,  # Disable evaluation if evaluation_strategy is unavailable
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save final model
model.save_pretrained("sql_bart_model")
tokenizer.save_pretrained("sql_bart_model")
