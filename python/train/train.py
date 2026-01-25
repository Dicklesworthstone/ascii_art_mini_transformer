"""
Training script for ASCII Art Transformer.

Implements:
- Mixed precision training (bfloat16/float16)
- Gradient accumulation
- Cosine learning rate schedule with warmup
- Checkpointing and resume
- Validation evaluation
- Console logging
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.transformer import AsciiGPT, AsciiGPTConfig, create_model
from model.tokenizer import get_tokenizer
from train.dataset import create_dataloaders, DataConfig


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Data
    db_path: str = "data/ascii_art.db"
    block_size: int = 2048

    # Model (will create from these if no model provided)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * this
    learning_rate: float = 6e-4
    max_iters: int = 100000
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 6e-5

    # Regularization
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Evaluation
    eval_interval: int = 500
    log_interval: int = 10
    eval_iters: int = 200

    # Checkpointing
    save_interval: int = 5000
    checkpoint_dir: str = "models/checkpoints"
    resume_from: Optional[str] = None

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"  # 'float32', 'float16', or 'bfloat16'
    compile_model: bool = False  # torch.compile (requires PyTorch 2.0+)

    # DataLoader
    num_workers: int = 4
    val_split: float = 0.1

    def __post_init__(self):
        """Validate and setup configuration."""
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Validate dtype
        if (
            self.device.startswith("cuda")
            and self.dtype == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            print("Warning: bfloat16 not supported, falling back to float16")
            self.dtype = "float16"

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def get_model_config(self) -> AsciiGPTConfig:
        """Create model config from training config."""
        return AsciiGPTConfig(
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
        )


def get_lr(iter_num: int, config: TrainingConfig) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.

    Args:
        iter_num: Current iteration number
        config: Training configuration

    Returns:
        Learning rate for this iteration
    """
    # Linear warmup
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters

    # After decay, use minimum
    if iter_num > config.lr_decay_iters:
        return config.min_lr

    # Cosine decay between warmup and decay iters
    decay_ratio = (iter_num - config.warmup_iters) / (
        config.lr_decay_iters - config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def _config_to_dict(config: TrainingConfig) -> dict:
    """Convert TrainingConfig to a plain dict for serialization."""
    return {
        "db_path": config.db_path,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "max_iters": config.max_iters,
        "warmup_iters": config.warmup_iters,
        "lr_decay_iters": config.lr_decay_iters,
        "min_lr": config.min_lr,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
    }


def _model_config_to_dict(model_config) -> dict | None:
    """Convert AsciiGPTConfig to a plain dict for serialization."""
    if model_config is None:
        return None
    return {
        "vocab_size": model_config.vocab_size,
        "block_size": model_config.block_size,
        "n_layer": model_config.n_layer,
        "n_head": model_config.n_head,
        "n_embd": model_config.n_embd,
        "dropout": model_config.dropout,
        "max_rows": model_config.max_rows,
        "max_cols": model_config.max_cols,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    best_val_loss: float,
    config: TrainingConfig,
    path: str | Path,
) -> None:
    """
    Save training checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer state
        iter_num: Current iteration
        best_val_loss: Best validation loss seen
        config: Training configuration
        path: Path to save checkpoint
    """
    # Save configs as plain dicts to avoid pickle module path issues
    model_config = model.config if hasattr(model, "config") else None

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "model_config": _model_config_to_dict(model_config),
        "training_config": _config_to_dict(config),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[int, float]:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to restore state

    Returns:
        Tuple of (iter_num, best_val_loss)
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Loaded checkpoint from {path} at iter {iter_num}")
    return iter_num, best_val_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    config: TrainingConfig,
    max_iters: Optional[int] = None,
) -> float:
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        config: Training configuration
        max_iters: Maximum number of batches to evaluate (None = all)

    Returns:
        Average validation loss
    """
    model.eval()
    losses = []
    max_iters = max_iters or config.eval_iters

    for i, batch in enumerate(val_loader):
        if i >= max_iters:
            break

        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)

        # Match training behavior: only autocast on CUDA when using a reduced-precision dtype.
        device_type = _get_device_type(config.device)
        use_amp = config.dtype != "float32" and device_type == "cuda"
        if use_amp:
            with torch.amp.autocast(
                device_type=device_type, dtype=_get_torch_dtype(config.dtype)
            ):
                _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
        else:
            _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)

        if loss is not None:
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def _get_device_type(device: str) -> str:
    """Return the torch device type string for AMP/autocast (e.g., 'cuda' for 'cuda:0')."""
    return torch.device(device).type


def train(
    config: TrainingConfig,
    model: Optional[AsciiGPT] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
) -> AsciiGPT:
    """
    Main training loop.

    Args:
        config: Training configuration
        model: Optional pre-initialized model
        train_loader: Optional pre-created train loader
        val_loader: Optional pre-created val loader

    Returns:
        Trained model
    """
    # Create tokenizer
    tokenizer = get_tokenizer()

    # Create model if not provided
    if model is None:
        model_config = config.get_model_config()
        model = create_model(model_config)

    # Move to device
    model = model.to(config.device)
    print(f"Model has {model.get_num_params():,} parameters")

    # Create data loaders if not provided
    if train_loader is None or val_loader is None:
        data_config = DataConfig(
            db_path=config.db_path,
            block_size=config.block_size,
        )
        train_loader, val_loader = create_dataloaders(
            config.db_path,
            tokenizer,
            batch_size=config.batch_size,
            val_split=config.val_split,
            num_workers=config.num_workers,
            config=data_config,
            pin_memory=config.device.startswith("cuda"),
        )
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        if len(train_loader) == 0:
            raise ValueError(
                "Train loader has 0 batches (likely batch_size too large for dataset). "
                "Reduce batch_size and/or val_split."
            )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Optional: compile model for speed
    if config.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Resume from checkpoint if specified
    iter_num = 0
    best_val_loss = float("inf")
    if config.resume_from:
        iter_num, best_val_loss = load_checkpoint(config.resume_from, model, optimizer)

    # Get dtype for autocast
    dtype = _get_torch_dtype(config.dtype)
    device_type = _get_device_type(config.device)
    use_amp = config.dtype != "float32" and device_type == "cuda"

    # Create gradient scaler for float16 (not needed for bfloat16)
    scaler = (
        torch.cuda.amp.GradScaler()
        if config.dtype == "float16" and device_type == "cuda"
        else None
    )

    # Training loop
    model.train()
    t0 = time.time()
    running_loss = 0.0

    # Create infinite iterator over training data
    def infinite_loader():
        while True:
            for batch in train_loader:
                yield batch

    data_iter = iter(infinite_loader())

    print(f"\nStarting training from iter {iter_num}")
    print(f"Device: {config.device}, dtype: {config.dtype}")
    print(f"Effective batch size: {config.effective_batch_size}")
    print("-" * 50)

    while iter_num < config.max_iters:
        # Update learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.gradient_accumulation_steps):
            batch = next(data_iter)

            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            # Forward pass with mixed precision
            if use_amp:
                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    _, loss = model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = loss / config.gradient_accumulation_steps
            else:
                _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss / config.gradient_accumulation_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()

        # Gradient clipping
        if scaler is not None:
            scaler.unscale_(optimizer)
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Logging
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            avg_loss = (
                running_loss / config.log_interval if iter_num > 0 else running_loss
            )
            running_loss = 0.0
            print(
                f"iter {iter_num:6d} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | {dt * 1000:.1f}ms"
            )

        # Evaluation
        if iter_num > 0 and iter_num % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, config)
            print(f">>> val loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    iter_num,
                    best_val_loss,
                    config,
                    Path(config.checkpoint_dir) / "best.pt",
                )

        # Periodic checkpointing
        if iter_num > 0 and iter_num % config.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                iter_num,
                best_val_loss,
                config,
                Path(config.checkpoint_dir) / f"ckpt_{iter_num}.pt",
            )

        iter_num += 1

    # Final save
    save_checkpoint(
        model,
        optimizer,
        iter_num,
        best_val_loss,
        config,
        Path(config.checkpoint_dir) / "final.pt",
    )

    return model


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train ASCII Art Transformer")

    # Data args
    parser.add_argument("--db-path", type=str, default="data/ascii_art.db")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)

    # Model args
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Model size preset (sets n_layer/n_head/n_embd/block_size defaults)",
    )
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training args
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--lr-decay-iters", type=int, default=100000)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Evaluation
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=500)

    # Hardware
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--num-workers", type=int, default=4)

    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Create a TrainingConfig from parsed CLI args."""
    from model.transformer import get_large_config, get_medium_config, get_small_config

    presets = {
        "small": get_small_config,
        "medium": get_medium_config,
        "large": get_large_config,
    }
    model_config = presets[args.preset]()
    block_size = (
        args.block_size if args.block_size is not None else model_config.block_size
    )
    n_layer = args.n_layer if args.n_layer is not None else model_config.n_layer
    n_head = args.n_head if args.n_head is not None else model_config.n_head
    n_embd = args.n_embd if args.n_embd is not None else model_config.n_embd

    return TrainingConfig(
        db_path=args.db_path,
        block_size=block_size,
        val_split=args.val_split,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=args.dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile,
        num_workers=args.num_workers,
    )


def parse_cli_config(argv: list[str] | None = None) -> TrainingConfig:
    """Parse CLI args into a TrainingConfig (does not start training)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return config_from_args(args)


def main() -> None:
    """CLI entry point."""
    config = parse_cli_config()
    train(config)


if __name__ == "__main__":
    main()
