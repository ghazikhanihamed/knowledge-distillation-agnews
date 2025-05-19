import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

# Set device
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Dataset loading and preprocessing
def load_and_preprocess_data(sample_fraction=1.0):
    print(f"Loading {sample_fraction*100}% of AG News dataset...")
    ag_news_dataset = load_dataset("ag_news")

    # Use a subset of the data for faster iterations
    if sample_fraction < 1.0:
        train_data = ag_news_dataset["train"].select(
            range(int(len(ag_news_dataset["train"]) * sample_fraction))
        )
        test_data = ag_news_dataset["test"].select(
            range(int(len(ag_news_dataset["test"]) * sample_fraction))
        )
    else:
        train_data = ag_news_dataset["train"]
        test_data = ag_news_dataset["test"]

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    return train_data, test_data


# Custom dataset for AG News
class AGNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(train_data, test_data, tokenizer, batch_size=64):
    train_dataset = AGNewsDataset(train_data, tokenizer)
    test_dataset = AGNewsDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


# 1. BASELINE: Fine-tune DistilBERT directly on AG News
def train_baseline_student(train_dataloader, test_dataloader, num_epochs=3):
    print("\n" + "=" * 50)
    print("Training Baseline Student Model (DistilBERT)")
    print("=" * 50)

    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    model.to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation phase
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average="weighted")

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")

    # Final evaluation
    print("\nFinal Evaluation:")
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average="weighted")

    report = classification_report(val_labels, val_preds)

    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

    results = {"model": model, "accuracy": accuracy, "f1_score": f1}

    return results


# 2. ENHANCED: Train Student with Knowledge Distillation
def train_enhanced_student(
    train_dataloader,
    test_dataloader,
    teacher_model,
    num_epochs=3,
    temperature=2.0,
    alpha=0.5,
):
    print("\n" + "=" * 50)
    print("Training Enhanced Student Model with Knowledge Distillation")
    print("=" * 50)

    # Initialize student model
    student_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    student_model.to(device)

    # Set teacher model to eval mode (frozen)
    teacher_model.to(device)
    teacher_model.eval()

    # Initialize optimizer
    optimizer = AdamW(student_model.parameters(), lr=2e-5)

    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Define distillation loss function
    def distillation_loss(student_logits, teacher_logits, labels, temp, alpha):
        # Soft targets loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / temp, dim=1)
        soft_log_probs = F.log_softmax(student_logits / temp, dim=1)
        soft_targets_loss = F.kl_div(
            soft_log_probs, soft_targets, reduction="batchmean"
        ) * (temp**2)

        # Hard targets loss (cross entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        loss = alpha * soft_targets_loss + (1 - alpha) * hard_loss
        return loss

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        student_model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

            # Get student predictions
            student_outputs = student_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            student_logits = student_outputs.logits

            # Calculate loss
            loss = distillation_loss(
                student_logits, teacher_logits, labels, temp=temperature, alpha=alpha
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation phase
        student_model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = student_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average="weighted")

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")

    # Final evaluation
    print("\nFinal Evaluation:")
    student_model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average="weighted")

    report = classification_report(val_labels, val_preds)

    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

    results = {"model": student_model, "accuracy": accuracy, "f1_score": f1}

    return results


# 3. Evaluate Teacher Model
def evaluate_teacher_model(test_dataloader, teacher_model):
    print("\n" + "=" * 50)
    print("Evaluating Teacher Model (BERT)")
    print("=" * 50)

    teacher_model.to(device)
    teacher_model.eval()

    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating Teacher"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average="weighted")

    report = classification_report(val_labels, val_preds)

    print(f"Teacher Accuracy: {accuracy:.4f}")
    print(f"Teacher F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

    results = {"model": teacher_model, "accuracy": accuracy, "f1_score": f1}

    return results


# 4. Compare model sizes
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def compare_models(teacher_results, baseline_results, enhanced_results):
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)

    teacher_params = count_parameters(teacher_results["model"])
    baseline_params = count_parameters(baseline_results["model"])
    enhanced_params = count_parameters(enhanced_results["model"])

    teacher_acc = teacher_results["accuracy"]
    baseline_acc = baseline_results["accuracy"]
    enhanced_acc = enhanced_results["accuracy"]

    teacher_f1 = teacher_results["f1_score"]
    baseline_f1 = baseline_results["f1_score"]
    enhanced_f1 = enhanced_results["f1_score"]

    # Create comparison table
    print("Model Parameter Counts:")
    print(f"Teacher (BERT): {teacher_params:,} parameters")
    print(f"Student (DistilBERT): {baseline_params:,} parameters")
    print(f"Parameter Reduction: {(1 - baseline_params/teacher_params)*100:.2f}%")

    print("\nPerformance Metrics:")
    print(f"{'Model':<25} {'Accuracy':<10} {'F1 Score':<10}")
    print(f"{'-'*45}")
    print(f"{'Teacher (BERT)':<25} {teacher_acc:.4f}    {teacher_f1:.4f}")
    print(f"{'Baseline Student':<25} {baseline_acc:.4f}    {baseline_f1:.4f}")
    print(f"{'Enhanced Student':<25} {enhanced_acc:.4f}    {enhanced_f1:.4f}")


def save_results_to_file(
    teacher_results,
    baseline_results,
    enhanced_results,
    filename="knowledge_distillation_results.txt",
):
    """Save the model comparison results to a text file."""
    with open(filename, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("Knowledge Distillation Results\n")
        f.write("=" * 50 + "\n\n")

        # Model parameters
        teacher_params = count_parameters(teacher_results["model"])
        baseline_params = count_parameters(baseline_results["model"])
        enhanced_params = count_parameters(enhanced_results["model"])

        f.write("Model Parameter Counts:\n")
        f.write(f"Teacher (BERT): {teacher_params:,} parameters\n")
        f.write(f"Student (DistilBERT): {baseline_params:,} parameters\n")
        f.write(
            f"Parameter Reduction: {(1 - baseline_params/teacher_params)*100:.2f}%\n\n"
        )

        # Performance metrics
        teacher_acc = teacher_results["accuracy"]
        baseline_acc = baseline_results["accuracy"]
        enhanced_acc = enhanced_results["accuracy"]

        teacher_f1 = teacher_results["f1_score"]
        baseline_f1 = baseline_results["f1_score"]
        enhanced_f1 = enhanced_results["f1_score"]

        f.write("Performance Metrics:\n")
        f.write(f"{'Model':<25} {'Accuracy':<10} {'F1 Score':<10}\n")
        f.write(f"{'-'*45}\n")
        f.write(f"{'Teacher (BERT)':<25} {teacher_acc:.4f}    {teacher_f1:.4f}\n")
        f.write(f"{'Baseline Student':<25} {baseline_acc:.4f}    {baseline_f1:.4f}\n")
        f.write(f"{'Enhanced Student':<25} {enhanced_acc:.4f}    {enhanced_f1:.4f}\n\n")

        f.write("\nFile generated on: " + time.strftime("%Y-%m-%d %H:%M:%S"))

    print(f"Results saved to {filename}")


# Main function
def main():
    print("Starting Knowledge Distillation Assignment")

    # 1. Load and preprocess data
    train_data, test_data = load_and_preprocess_data()

    # 2. Set up tokenizer and dataloaders
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataloader, test_dataloader = create_dataloaders(
        train_data, test_data, tokenizer
    )

    # 3. Load pre-trained teacher model
    print("\nLoading pre-trained teacher model...")
    teacher_model = BertForSequenceClassification.from_pretrained(
        "fabriceyhc/bert-base-uncased-ag_news"
    )
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 4. Train baseline student model
    baseline_results = train_baseline_student(
        train_dataloader, test_dataloader, num_epochs=3
    )

    # 5. Train enhanced student model with knowledge distillation
    enhanced_results = train_enhanced_student(
        train_dataloader,
        test_dataloader,
        teacher_model,
        num_epochs=3,
        temperature=2.0,  # Temperature for softening logits
        alpha=0.5,  # Weight for soft targets vs hard targets
    )

    # 6. Evaluate teacher model
    teacher_results = evaluate_teacher_model(test_dataloader, teacher_model)

    # 7. Compare models
    compare_models(teacher_results, baseline_results, enhanced_results)

    # 8. Save results to text file
    save_results_to_file(teacher_results, baseline_results, enhanced_results)

    print("\nKnowledge Distillation Assignment Complete!")


if __name__ == "__main__":
    main()
