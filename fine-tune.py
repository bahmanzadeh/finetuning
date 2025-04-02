import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
    DataCollatorWithPadding
)
from tqdm.auto import tqdm


def setup(rank, world_size):
    """Initialize the process group for distributed training."""
    local_rank = int(os.environ["LOCAL_RANK"])  # Local rank (GPU index on the current node)

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set the device to the local rank
    torch.cuda.set_device(local_rank)

    # Debug information
    print(f"[Rank {rank}] Process group initialized.")
    print(f"[Rank {rank}] Backend: {dist.get_backend()}")
    print(f"[Rank {rank}] World Size: {dist.get_world_size()}")
    print(f"[Rank {rank}] Master Address: {os.environ.get('MASTER_ADDR', 'Not Set')}")
    print(f"[Rank {rank}] Master Port: {os.environ.get('MASTER_PORT', 'Not Set')}")
    print(f"[Rank {rank}] Current Device: {torch.cuda.current_device()}")


def cleanup():
    """Destroy the process group after training."""
    dist.destroy_process_group()


def load_model_with_barrier(model_name, num_labels=2, ignore_mismatched_sizes=True):
    """Load a model with a distributed barrier to avoid concurrent downloads."""
    os.environ["TRANSFORMERS_NO_ADVISORY_LOCKS"] = "1"

    if dist.get_rank() == 0:
        # Rank 0 downloads the model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes
        )
    dist.barrier()  # Synchronize all processes
    if dist.get_rank() != 0:
        # Other ranks load the model after synchronization
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes
        )
    return model


def train(rank, world_size, args):
    """Main training function for each process."""
    # Setup distributed training
    setup(rank, world_size)

    # Debug: Print rank-specific information
    print(f"[Rank {rank}] Starting training with world size {world_size}.")
    local_rank = int(os.environ["LOCAL_RANK"])
    # Load dataset
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(args.train_dataset_range))
    eval_dataset = dataset["validation"].select(range(args.eval_dataset_range))

    # Load tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model using the barrier function
    # model = load_model_with_barrier(model_name).to(rank)
    
    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(local_rank)

    # Preprocess datasets
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Distributed sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Data loaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=data_collator)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Shuffle data for each epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Global Rank {rank}]", disable=rank != 0)
        for batch in progress_bar:
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.set_postfix({"loss": loss.item()})

        # Evaluation loop
        model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(local_rank) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                labels = batch["labels"]
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels, reduction="sum")
                eval_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        eval_loss /= total
        accuracy = correct / total
        if rank == 0:
            print(f"Epoch {epoch + 1}: Eval Loss = {eval_loss:.4f}, Accuracy = {accuracy:.4f}")

    # Save the model (only on rank 0)
    if rank == 0:
        output_dir = os.path.join(args.shared_fs_path, args.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        model.module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Training complete!")

    cleanup()


def main():
    # Read environment variables set by torchrun
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes
    rank = int(os.environ["RANK"])  # Global rank of the current process
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank of the process on the current node

    print(f"Starting training with world size {world_size}, rank {rank}, and local rank {local_rank}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default="default-experiment", help="Name for this experiment")
    parser.add_argument("--shared_fs_path", type=str, default="/shared", help="Path to shared filesystem")
    parser.add_argument("--train_dataset_range", type=int, default=1000, help="Number of training samples to use")
    parser.add_argument("--eval_dataset_range", type=int, default=200, help="Number of evaluation samples to use")
    args = parser.parse_args()

    train(rank, world_size, args)

    # Spawn processes for distributed training
    #mp.spawn(train, args=(world_size, args), nprocs=int(os.environ["NPROC_PER_NODE"]), join=True)


if __name__ == "__main__":
    main()
