"""ml/pytorch/train.py — minimal PyTorch training loop skeleton.

Accelerate-compatible, WandB + MLflow dual logging, mixed precision from .env.
Fill in dataset / model / loss for the actual task. Do NOT commit real data.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Optional heavy deps loaded lazily so this file imports on a bare env for tests.
try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None  # type: ignore

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore


def build_model():
    """TODO: replace with real architecture (timm.create_model, HF AutoModel, ...)."""
    return torch.nn.Linear(16, 2)


def build_dataloaders(batch_size: int):
    """TODO: return (train_loader, val_loader). Placeholder returns random data."""
    x = torch.randn(128, 16)
    y = torch.randint(0, 2, (128,))
    ds = torch.utils.data.TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), None


def main():
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    epochs = int(os.getenv("EPOCHS", "1"))
    lr = float(os.getenv("LR", "1e-3"))
    mixed = os.getenv("MIXED_PRECISION", "fp16")  # "no" | "fp16" | "bf16"

    accelerator = Accelerator(mixed_precision=mixed) if Accelerator else None
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader, _ = build_dataloaders(batch_size)

    if accelerator:
        model, optim, train_loader = accelerator.prepare(model, optim, train_loader)

    # Tracking
    if wandb and os.getenv("WANDB_API_KEY"):
        wandb.init(project="hacknation-2026", config={"lr": lr, "bs": batch_size, "epochs": epochs})
    if mlflow:
        mlflow.set_tracking_uri("file:./experiments/mlruns")
        mlflow.set_experiment("pytorch-baseline")
        mlflow.start_run()
        mlflow.log_params({"lr": lr, "bs": batch_size, "epochs": epochs})

    for epoch in range(epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            if accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()
            optim.step()

            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")
                if wandb and wandb.run:
                    wandb.log({"loss": loss.item(), "epoch": epoch, "step": step})
                if mlflow:
                    mlflow.log_metric("loss", loss.item(), step=epoch * len(train_loader) + step)

    ckpt_dir = Path("experiments/pytorch-ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "last.pt")

    if wandb and wandb.run:
        wandb.finish()
    if mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
