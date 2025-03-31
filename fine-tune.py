import argparse
import os
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import torch
from accelerate import Accelerator

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metric = evaluate.load("accuracy")
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    return acc

def main(args):
    # Initialize Accelerator
    accelerator = Accelerator()

    # Log node_rank and local_rank
    node_rank = int(os.environ.get("NODE_RANK", 0))  # Get NODE_RANK from environment
    local_rank = accelerator.local_process_index  # Get local rank from Accelerator
    print(f"Node Rank: {node_rank}, Local Rank: {local_rank}")

    # Set up logging directory on shared filesystem
    output_dir = os.path.join(args.shared_fs_path, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Validate Hugging Face Hub arguments
    if args.push_to_hub and (not args.hub_model_id or not args.hub_token):
        raise ValueError("Both --hub_model_id and --hub_token are required when --push_to_hub is set.")

    # Load dataset (SST-2 from GLUE)
    try:
        dataset = load_dataset("glue", "sst2")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Use a small subset for demonstration
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = dataset["validation"].select(range(200))

    # Load pre-trained model and tokenizer (DistilBERT)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Preprocess function for text classification
    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=128)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # Prepare model, datasets, and optimizer with Accelerator
    model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",  # Replace evaluation_strategy with eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_token=args.hub_token if args.push_to_hub else None,
        report_to="tensorboard",
    )

    # Create a data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Fine-tune model
    print("Starting fine-tuning...")
    trainer.train()
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print("Fine-tuned metrics:", metrics)

    # Save and push the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default="default-experiment", help="Name for this experiment (used for output directory)")
    parser.add_argument("--shared_fs_path", type=str, default="/shared", help="Path to shared filesystem (PVC mount)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, help="Model ID for Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--hub_token", type=str, help="Hugging Face Hub token")
    args = parser.parse_args()
    main(args)
