import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import evaluate
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from transformers import logging as transformers_logging

# Fix progress bar
transformers_logging.set_verbosity_info()

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define settings for multiple runs
settings = [
    {  # Run 1:
        "name": "run1",
        "lr_enc": 3e-5,
        "lr_dec": 2e-5,
        "temp": 2.0,
        "alpha": 0.5,
        "epochs_enc": 2,
        "epochs_dec": 2,
        "batch_size": 64,
    },
    {  # Run 2:
        "name": "run2",
        "lr_enc": 3e-5,
        "lr_dec": 2e-5,
        "temp": 5.0,
        "alpha": 0.3,
        "epochs_enc": 2,
        "epochs_dec": 2,
        "batch_size": 64,
    },
    {  # Run 3:
        "name": "run3",
        "lr_enc": 1e-5,
        "lr_dec": 1e-5,
        "temp": 2.0,
        "alpha": 0.7,
        "epochs_enc": 2,
        "epochs_dec": 2,
        "batch_size": 64,
    },
    {  # Run 4:
        "name": "run4",
        "lr_enc": 3e-5,
        "lr_dec": 2e-5,
        "temp": 9.0,
        "alpha": 0.8,
        "epochs_enc": 2,
        "epochs_dec": 2,
        "batch_size": 32,
    },
]

# ======================== CHECK MAX LENGTH
dataset = load_dataset("ag_news", split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

lengths = []
for text in dataset["text"]:
    toks = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    lengths.append(len(toks))

percentiles = [50, 90, 95, 99, 100]
values = np.percentile(lengths, percentiles).astype(int)

df = pd.DataFrame({"percentile": percentiles, "token_length": values})
print(df)
# ======================== DATA PREPARATION

print("\n Loading AG News dataset...")
ag_news_dataset = load_dataset("ag_news")

train_data = ag_news_dataset["train"]
test_data = ag_news_dataset["test"]

print(f" {len(train_data)} training examples, {len(test_data)} test examples")
print(f" Classes: {ag_news_dataset['train'].features['label'].names}")

split = train_data.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]
print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

# ======================== MODEL SETUP

print("\n Loading models...")
teacher_model_name = "fabriceyhc/bert-base-uncased-ag_news"
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    teacher_model_name
).to(device)
print(f" Teacher model loaded: {teacher_model_name}")

student_model_name = "distilbert-base-uncased"
student_model = AutoModelForSequenceClassification.from_pretrained(
    student_model_name, num_labels=4
).to(device)
print(f" Student model loaded: {student_model_name}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ======================== DATA PROCESSING

print("\n Processing dataset...")


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=64
    )


# train_data = train_data.map(tokenize_function, batched=True)
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)
# train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print("Data processed and formatted for PyTorch")

# ======================== DECODER-ONLY DATA

decoder_student_name = "distilgpt2"
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_student_name)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token


def decoder_tokenize_function(examples):
    return decoder_tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=64
    )


print("Tokenizing dataset for decoder-only model...")
decoder_train_data = ag_news_dataset["train"].map(
    decoder_tokenize_function, batched=True
)
decoder_test_data = ag_news_dataset["test"].map(decoder_tokenize_function, batched=True)
decoder_train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
decoder_test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

split = decoder_train_data.train_test_split(test_size=0.1, seed=42)
decoder_train_dataset = split["train"]
decoder_val_dataset = split["test"]

# ======================== TRAINING SETUP


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = evaluate.load("accuracy").compute(
        predictions=predictions, references=labels
    )["accuracy"]
    f1 = evaluate.load("f1").compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]
    return {"accuracy": accuracy, "f1": f1}


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits
        loss_ce = F.cross_entropy(student_logits.view(-1, 4), labels.view(-1))
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
        ) * (self.temperature**2)
        loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kl
        return (loss, outputs) if return_outputs else loss


# ======================== RUN EXPERIMENTS

for setting in settings:
    run_name = setting["name"]
    results_file = f"results_summary_{run_name}.txt"
    print(f"\nStarting Run: {run_name}")
    with open(results_file, "w") as f:
        f.write(f"Results Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Settings: {setting}\n")
        f.write(
            f"{len(train_data)} training examples, {len(test_data)} test examples\n"
        )
        f.write(f"Classes: {ag_news_dataset['train'].features['label'].names}\n")
        f.write(f"Teacher model: {teacher_model_name}\n")
        f.write(f"Student model: {student_model_name}\n")

    def write_to_results(content):
        with open(results_file, "a") as f:
            f.write(content + "\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results_{run_name}",
        eval_strategy="epoch",
        learning_rate=setting["lr_enc"],
        per_device_train_batch_size=setting["batch_size"],
        per_device_eval_batch_size=setting["batch_size"],
        num_train_epochs=setting["epochs_enc"],
        weight_decay=0.01,
        logging_dir=f"./logs_{run_name}",
        logging_steps=200,
        report_to="none",
        dataloader_pin_memory=False,
        save_strategy="epoch",
        seed=42,
        save_total_limit=1,
        warmup_steps=500,
        load_best_model_at_end=True,
    )

    decoder_training_args = TrainingArguments(
        output_dir=f"./results_decoder_{run_name}",
        eval_strategy="epoch",
        learning_rate=setting["lr_dec"],
        per_device_train_batch_size=setting["batch_size"],
        per_device_eval_batch_size=setting["batch_size"],
        num_train_epochs=setting["epochs_dec"],
        weight_decay=0.01,
        logging_dir=f"./logs_decoder_{run_name}",
        logging_steps=200,
        report_to="none",
        dataloader_pin_memory=False,
        save_strategy="epoch",
        seed=42,
        save_total_limit=1,
        warmup_steps=500,
        load_best_model_at_end=True,
    )

    # EXPERIMENT 1: Baseline student model
    print("\n EXPERIMENT 1: Training baseline student model...")
    write_to_results("\nEXPERIMENT 1: Training baseline student model...")
    baseline_student = AutoModelForSequenceClassification.from_pretrained(
        student_model_name, num_labels=4
    ).to(device)

    baseline_trainer = Trainer(
        model=baseline_student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    start_time = time.time()
    baseline_trainer.train()
    baseline_time = time.time() - start_time
    baseline_results = baseline_trainer.evaluate(test_data)
    print(f"Baseline student trained in {baseline_time:.1f} seconds")
    print(
        f"Accuracy: {baseline_results['eval_accuracy']:.4f}, F1: {baseline_results['eval_f1']:.4f}"
    )
    write_to_results(f"Baseline student trained in {baseline_time:.1f} seconds")
    write_to_results(
        f"Accuracy: {baseline_results['eval_accuracy']:.4f}, F1: {baseline_results['eval_f1']:.4f}"
    )

    # EXPERIMENT 2: Distilled student model
    print("\n EXPERIMENT 2: Training student with knowledge distillation...")
    write_to_results("\nEXPERIMENT 2: Training student with knowledge distillation...")
    distill_student = AutoModelForSequenceClassification.from_pretrained(
        student_model_name, num_labels=4
    ).to(device)

    distill_trainer = DistillationTrainer(
        model=distill_student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        teacher_model=teacher_model,
        temperature=setting["temp"],
        alpha=setting["alpha"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    start_time = time.time()
    distill_trainer.train()
    distill_time = time.time() - start_time
    distill_results = distill_trainer.evaluate(test_data)
    print(f"Distilled student trained in {distill_time:.1f} seconds")
    print(
        f"Accuracy: {distill_results['eval_accuracy']:.4f}, F1: {distill_results['eval_f1']:.4f}"
    )
    write_to_results(f"Distilled student trained in {distill_time:.1f} seconds")
    write_to_results(
        f"Accuracy: {distill_results['eval_accuracy']:.4f}, F1: {distill_results['eval_f1']:.4f}"
    )

    # Teacher evaluation
    print("\n Evaluating teacher model...")
    write_to_results("\nEvaluating teacher model...")
    teacher_trainer = Trainer(
        model=teacher_model,
        args=training_args,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )
    teacher_results = teacher_trainer.evaluate()
    print(
        f"Teacher accuracy: {teacher_results['eval_accuracy']:.4f}, F1: {teacher_results['eval_f1']:.4f}"
    )
    write_to_results(
        f"Teacher accuracy: {teacher_results['eval_accuracy']:.4f}, F1: {teacher_results['eval_f1']:.4f}"
    )

    # EXPERIMENT 3: Decoder-only student
    print("\n EXPERIMENT 3: Distillation with Decoder-Only Student...")
    write_to_results("\nEXPERIMENT 3: Distillation with Decoder-Only Student...")
    decoder_student = AutoModelForSequenceClassification.from_pretrained(
        decoder_student_name, num_labels=4
    ).to(device)
    decoder_student.config.pad_token_id = decoder_tokenizer.pad_token_id

    # Pre-fine-tune decoder
    print("Pre-fine-tuning decoder-only student...")
    write_to_results("Pre-fine-tuning decoder-only student...")
    pretrain_trainer = Trainer(
        model=decoder_student,
        args=decoder_training_args,
        train_dataset=decoder_train_dataset,
        eval_dataset=decoder_val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    pretrain_trainer.train()
    pretrain_results = pretrain_trainer.evaluate(decoder_test_data)
    print(
        f"Pre-fine-tuned decoder accuracy: {pretrain_results['eval_accuracy']:.4f}, F1: {pretrain_results['eval_f1']:.4f}"
    )
    write_to_results(
        f"Pre-fine-tuned decoder accuracy: {pretrain_results['eval_accuracy']:.4f}, F1: {pretrain_results['eval_f1']:.4f}"
    )

    # Distillation
    print("Training decoder-only student with knowledge distillation...")
    write_to_results("Training decoder-only student with knowledge distillation...")
    decoder_distillation_trainer = DistillationTrainer(
        model=decoder_student,
        args=decoder_training_args,
        train_dataset=decoder_train_dataset,
        eval_dataset=decoder_val_dataset,
        compute_metrics=compute_metrics,
        teacher_model=teacher_model,
        temperature=setting["temp"],
        alpha=setting["alpha"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    start_time = time.time()
    decoder_distillation_trainer.train()
    decoder_time = time.time() - start_time
    decoder_results = decoder_distillation_trainer.evaluate(decoder_test_data)
    print(f"Decoder-only student trained in {decoder_time:.1f} seconds")
    print(
        f"Accuracy: {decoder_results['eval_accuracy']:.4f}, F1: {decoder_results['eval_f1']:.4f}"
    )
    write_to_results(f"Decoder-only student trained in {decoder_time:.1f} seconds")
    write_to_results(
        f"Accuracy: {decoder_results['eval_accuracy']:.4f}, F1: {decoder_results['eval_f1']:.4f}"
    )

    # Results analysis
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in distill_student.parameters())
    decoder_params = sum(p.numel() for p in decoder_student.parameters())
    compression_ratio = teacher_params / student_params

    print("\n MODEL COMPARISON")
    print("=" * 70)
    print(
        f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Parameters':<12} {'Time (s)':<10}"
    )
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    print(
        f"{'BERT (Teacher)':<20} {teacher_results['eval_accuracy']:.4f}     {teacher_results['eval_f1']:.4f}     {teacher_params:,}  {'N/A':<10}"
    )
    print(
        f"{'DistilBERT':<20} {baseline_results['eval_accuracy']:.4f}     {baseline_results['eval_f1']:.4f}     {student_params:,}  {baseline_time:.1f}"
    )
    print(
        f"{'DistilBERT + KD':<20} {distill_results['eval_accuracy']:.4f}     {distill_results['eval_f1']:.4f}     {student_params:,}  {distill_time:.1f}"
    )
    print(
        f"{'DistilGPT2 + KD':<20} {decoder_results['eval_accuracy']:.4f}     {decoder_results['eval_f1']:.4f}     {decoder_params:,}  {decoder_time:.1f}"
    )
    print("=" * 70)

    write_to_results("\nMODEL COMPARISON")
    write_to_results("=" * 70)
    write_to_results(
        f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Parameters':<12} {'Time (s)':<10}"
    )
    write_to_results(f"{'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    write_to_results(
        f"{'BERT (Teacher)':<20} {teacher_results['eval_accuracy']:.4f}     {teacher_results['eval_f1']:.4f}     {teacher_params:,}  {'N/A':<10}"
    )
    write_to_results(
        f"{'DistilBERT':<20} {baseline_results['eval_accuracy']:.4f}     {baseline_results['eval_f1']:.4f}     {student_params:,}  {baseline_time:.1f}"
    )
    write_to_results(
        f"{'DistilBERT + KD':<20} {distill_results['eval_accuracy']:.4f}     {distill_results['eval_f1']:.4f}     {student_params:,}  {distill_time:.1f}"
    )
    write_to_results(
        f"{'DistilGPT2 + KD':<20} {decoder_results['eval_accuracy']:.4f}     {decoder_results['eval_f1']:.4f}     {decoder_params:,}  {decoder_time:.1f}"
    )
    write_to_results("=" * 70)

    acc_improvement = (
        distill_results["eval_accuracy"] - baseline_results["eval_accuracy"]
    )
    f1_improvement = distill_results["eval_f1"] - baseline_results["eval_f1"]

    print(f"\n IMPROVEMENT FROM KNOWLEDGE DISTILLATION:")
    print(f"Accuracy: +{acc_improvement:.4f} absolute (+{acc_improvement*100:.2f}%)")
    print(f"F1 Score: +{f1_improvement:.4f} absolute (+{f1_improvement*100:.2f}%)")
    print(f"Compression: {compression_ratio:.1f}x smaller than teacher")
    print(
        f"Performance: {distill_results['eval_accuracy']/teacher_results['eval_accuracy']*100:.1f}% of teacher accuracy with only {student_params/teacher_params*100:.1f}% of parameters"
    )

    write_to_results("\nIMPROVEMENT FROM KNOWLEDGE DISTILLATION:")
    write_to_results(
        f"Accuracy: +{acc_improvement:.4f} absolute (+{acc_improvement*100:.2f}%)"
    )
    write_to_results(
        f"F1 Score: +{f1_improvement:.4f} absolute (+{f1_improvement*100:.2f}%)"
    )
    write_to_results(f"Compression: {compression_ratio:.1f}x smaller than teacher")
    write_to_results(
        f"Performance: {distill_results['eval_accuracy']/teacher_results['eval_accuracy']*100:.1f}% of teacher accuracy with only {student_params/teacher_params*100:.1f}% of parameters"
    )

    # Visualization
    models = ["BERT (Teacher)", "DistilBERT", "DistilBERT + KD", "DistilGPT2 + KD"]
    accuracies = [
        teacher_results["eval_accuracy"],
        baseline_results["eval_accuracy"],
        distill_results["eval_accuracy"],
        decoder_results["eval_accuracy"],
    ]
    f1_scores = [
        teacher_results["eval_f1"],
        baseline_results["eval_f1"],
        distill_results["eval_f1"],
        decoder_results["eval_f1"],
    ]
    params = [
        teacher_params / 1_000_000,
        student_params / 1_000_000,
        student_params / 1_000_000,
        decoder_params / 1_000_000,
    ]

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    x = range(len(models))
    width = 0.35
    plt.bar([i - width / 2 for i in x], accuracies, width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], f1_scores, width, label="F1 Score")
    plt.ylim(0, 1.0)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title(f"Performance Comparison ({run_name})")
    plt.xticks(x, models, rotation=15)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.bar(models, params, color=["darkblue", "lightblue", "green", "orange"])
    plt.xlabel("Model")
    plt.ylabel("Parameters (millions)")
    plt.title("Model Size Comparison")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    times = [
        "N/A",
        f"{baseline_time:.1f}s",
        f"{distill_time:.1f}s",
        f"{decoder_time:.1f}s",
    ]
    for i, (p, t) in enumerate(zip(params, times)):
        plt.text(i, p + 0.5, t, ha="center")

    plt.tight_layout()
    plt.savefig(f"distillation_results_{run_name}.png", dpi=100)
    plt.close()

    # Save artifact
    with open(results_file, "r") as f:
        results_content = f.read()
    artifact_content = f""" {results_content} """
    with open(f"artifact_{run_name}.txt", "w") as f:
        f.write(artifact_content)

if __name__ == "__main__":
    print("Done.")
