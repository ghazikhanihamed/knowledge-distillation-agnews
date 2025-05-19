import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import evaluate
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("\nLoading AG News dataset...")
ag_news_dataset = load_dataset("ag_news")

train_data = ag_news_dataset["train"]
test_data = ag_news_dataset["test"]
print(f"{len(train_data)} training examples, {len(test_data)} test examples")
print(f"Classes: {ag_news_dataset['train'].features['label'].names}")

split = train_data.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]
print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

print("\nLoading model and tokenizer...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(
    device
)
print(f"Model loaded: {model_name}")

print("\nProcessing dataset...")


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=64
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_data.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print("Data processed and formatted for PyTorch")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = evaluate.load("accuracy").compute(
        predictions=predictions, references=labels
    )["accuracy"]
    f1 = evaluate.load("f1").compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]
    return {"accuracy": accuracy, "f1": f1}


training_args = TrainingArguments(
    output_dir="./results_bert_teacher",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_bert_teacher",
    logging_steps=200,
    save_strategy="epoch",
    save_total_limit=1,
    seed=42,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\nFine-tuning BERT teacher model...")
trainer.train()

print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test F1 Score: {test_results['eval_f1']:.4f}")

print("\nSaving fine-tuned model and tokenizer...")
model.save_pretrained("./bert_teacher_ag_news")
tokenizer.save_pretrained("./bert_teacher_ag_news")
print("Model and tokenizer saved to ./bert_teacher_ag_news")

if __name__ == "__main__":
    print("Fine-tuning complete.")
