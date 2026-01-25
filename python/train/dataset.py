"""
Training data pipeline and DataLoader for ASCII art generation.

This module provides:
- AsciiArtDataset: PyTorch Dataset that loads from SQLite
- Collate function for variable-length batching
- Train/val split utilities
- DataLoader factory functions

The pipeline handles:
- Character-level tokenization
- Constraint conditioning (width/height)
- Efficient batching with padding
"""

from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Import from sibling packages
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.tokenizer import AsciiTokenizer, get_tokenizer
from train.augmentation import AugmentationConfig, augment_art, validate_augmented_art


@dataclass
class DataConfig:
    """Configuration for the training data pipeline."""

    db_path: str
    block_size: int = 2048
    add_constraints: bool = True
    constraint_prob: float = 0.8  # Probability of adding constraints
    width_prob: float = 0.5  # Probability of including width (given constraints)
    height_prob: float = 0.5  # Probability of including height (given constraints)
    charset: str = "ascii"  # 'ascii', 'extended', or 'unicode'
    min_chars: int = 10  # Minimum art size
    max_chars: Optional[int] = None  # Maximum art size (None = block_size - 50)
    augment: bool = False
    augment_prob: float = 0.5


def _compute_width_height(raw_text: str) -> tuple[int, int]:
    lines = raw_text.split("\n")
    if raw_text.endswith("\n") and lines and lines[-1] == "":
        lines = lines[:-1]
    height = len(lines)
    width = max((len(line) for line in lines), default=0)
    return width, height


class AsciiArtDataset(Dataset):
    """
    PyTorch Dataset for ASCII art training data.

    Loads art from SQLite database, tokenizes with a character-level tokenizer,
    and optionally adds constraint conditioning. 2D positions are derived from
    the newline token inside the model's positional encoding module.

    Args:
        db_path: Path to SQLite database
        tokenizer: AsciiTokenizer instance
        config: DataConfig with pipeline settings
    """

    def __init__(
        self,
        db_path: str | Path,
        tokenizer: Optional[AsciiTokenizer] = None,
        config: Optional[DataConfig] = None,
    ):
        self.db_path = str(db_path)
        self.tokenizer = tokenizer or get_tokenizer()
        self.config = config or DataConfig(db_path=self.db_path)
        self._conn: sqlite3.Connection | None = None
        self._conn_pid: int | None = None

        # Compute max chars if not specified
        if self.config.max_chars is None:
            # Leave room for constraints prefix (~50 tokens max)
            self.config.max_chars = self.config.block_size - 50

        # Load valid IDs from database
        self.ids = self._load_valid_ids()

    def __getstate__(self) -> dict:
        # sqlite3.Connection is not pickleable. Ensure DataLoader worker processes can
        # serialize the dataset even if it was used before for debugging.
        state = self.__dict__.copy()
        state["_conn"] = None
        state["_conn_pid"] = None
        return state

    def close(self) -> None:
        """Close any cached DB connection (best-effort)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._conn_pid = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _get_conn(self) -> sqlite3.Connection:
        """Get (or lazily create) a cached read-only connection for faster iteration."""
        pid = os.getpid()
        if self._conn is None or self._conn_pid != pid:
            if self._conn is not None:
                self._conn.close()
            conn = sqlite3.connect(self.db_path)
            # Defensive: prevent accidental writes from training code.
            try:
                conn.execute("PRAGMA query_only = 1;")
            except sqlite3.OperationalError:  # pragma: no cover
                # Some SQLite builds may not support this pragma; training is read-only anyway.
                pass
            self._conn = conn
            self._conn_pid = pid
        return self._conn

    def _load_valid_ids(self) -> list[int]:
        """Load IDs of valid art pieces that fit within block_size."""
        max_chars = self.config.max_chars
        if max_chars is None:  # pragma: no cover
            raise ValueError("config.max_chars must be set before loading IDs")

        def has_column(conn: sqlite3.Connection, *, table: str, column: str) -> bool:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            names = {str(row[1]).lower() for row in cursor.fetchall()}
            return column.lower() in names

        with sqlite3.connect(self.db_path) as conn:
            # Prefer charset filter when supported, but tolerate older schemas that lack it.
            ids: list[int] = []
            if has_column(conn, table="ascii_art", column="charset"):
                cursor = conn.execute(
                    """
                    SELECT id FROM ascii_art
                    WHERE is_valid = 1
                    AND charset = ?
                    AND total_chars >= ?
                    AND total_chars <= ?
                """,
                    (
                        self.config.charset,
                        self.config.min_chars,
                        max_chars,
                    ),
                )
                ids = [row[0] for row in cursor.fetchall()]

            if not ids:
                cursor = conn.execute(
                    """
                    SELECT id FROM ascii_art
                    WHERE is_valid = 1
                    AND total_chars >= ?
                    AND total_chars <= ?
                """,
                    (
                        self.config.min_chars,
                        max_chars,
                    ),
                )
                ids = [row[0] for row in cursor.fetchall()]

            return ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary with:
            - input_ids: Token IDs for the full example sequence
            - labels: Same as input_ids (the model does internal shifting for loss)
        """
        art_id = self.ids[idx]

        # Load from database
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT raw_text, width, height, description, category
            FROM ascii_art WHERE id = ?
        """,
            (art_id,),
        )
        row = cursor.fetchone()

        if row is None:
            raise IndexError(f"Art ID {art_id} not found in database")

        raw_text, width, height, description, category = row

        # Use description if available, otherwise use category or placeholder
        if not description:
            description = category if category else "ASCII art"

        # Determine style based on category
        style = self._infer_style(category, raw_text)

        # Decide whether to add constraints
        add_constraints = (
            self.config.add_constraints
            and random.random() < self.config.constraint_prob
        )

        if add_constraints:
            # Randomly include width and/or height
            include_width = random.random() < self.config.width_prob
            include_height = random.random() < self.config.height_prob

            tokens = self.tokenizer.encode_training_example(
                description=description,
                art=raw_text,
                width=width if include_width else None,
                height=height if include_height else None,
                style=style,
            )
        else:
            tokens = self.tokenizer.encode_training_example(
                description=description,
                art=raw_text,
                style=style,
            )

        # Truncate if too long
        if len(tokens) > self.config.block_size:
            tokens = tokens[: self.config.block_size]

        # Model expects labels = input_ids (it does internal shifting for loss)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _infer_style(self, category: Optional[str], raw_text: str) -> str:
        """Infer art style from category and content."""
        if not category:
            # Check if it looks like a FIGlet banner
            lines = raw_text.strip().split("\n")
            if len(lines) <= 10 and len(lines) >= 2:
                # Short height, might be banner
                avg_density = sum(
                    sum(1 for c in line if c not in " \t") / max(1, len(line))
                    for line in lines
                ) / len(lines)
                if avg_density > 0.3:
                    return "banner"
            return "art"

        category_lower = category.lower()
        if "banner" in category_lower or "figlet" in category_lower:
            return "banner"
        elif "simple" in category_lower or "line" in category_lower:
            return "simple"
        elif "detailed" in category_lower or "shade" in category_lower:
            return "detailed"
        else:
            return "art"


class AugmentedAsciiArtDataset(Dataset):
    """
    Wrapper dataset that applies data augmentation to training examples.

    Wraps an AsciiArtDataset and applies augmentations during __getitem__.
    Useful for training to improve model robustness.

    Args:
        base_dataset: The underlying AsciiArtDataset
        augmentation_config: Configuration for augmentations (None for defaults)
        augment_prob: Probability of applying any augmentation (0-1)
    """

    def __init__(
        self,
        base_dataset: AsciiArtDataset,
        augmentation_config: AugmentationConfig | None = None,
        augment_prob: float = 0.5,
    ):
        self.base_dataset = base_dataset
        self.augment_prob = augment_prob
        self.augmentation_config = augmentation_config or AugmentationConfig()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get an item with optional augmentation.

        Note: This re-fetches from the database and applies augmentation
        before tokenization. This is less efficient than the base dataset
        but necessary for augmentation to work properly.
        """
        # Check if we should augment
        if random.random() > self.augment_prob:
            return self.base_dataset[idx]

        # Get the art ID and fetch raw data
        art_id = self.base_dataset.ids[idx]
        conn = self.base_dataset._get_conn()
        cursor = conn.execute(
            """
            SELECT raw_text, width, height, description, category
            FROM ascii_art WHERE id = ?
        """,
            (art_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return self.base_dataset[idx]

        raw_text, width, height, description, category = row

        if not description:
            description = category if category else "ASCII art"

        # Apply augmentation
        augmented_art, augmented_desc = augment_art(
            raw_text, description, self.augmentation_config
        )
        if not validate_augmented_art(raw_text, augmented_art):
            return self.base_dataset[idx]

        # Re-tokenize with augmented data
        style = self.base_dataset._infer_style(category, augmented_art)

        # Decide whether to add constraints (reuse base config settings)
        config = self.base_dataset.config
        add_constraints = (
            config.add_constraints and random.random() < config.constraint_prob
        )

        tokenizer = self.base_dataset.tokenizer

        if add_constraints:
            include_width = random.random() < config.width_prob
            include_height = random.random() < config.height_prob

            # Recalculate dimensions for augmented art
            aug_width, aug_height = _compute_width_height(augmented_art)

            tokens = tokenizer.encode_training_example(
                description=augmented_desc,
                art=augmented_art,
                width=aug_width if include_width else None,
                height=aug_height if include_height else None,
                style=style,
            )
        else:
            tokens = tokenizer.encode_training_example(
                description=augmented_desc,
                art=augmented_art,
                style=style,
            )

        # Truncate if too long
        if len(tokens) > config.block_size:
            tokens = tokens[: config.block_size]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Collate function for variable-length sequences.

    Pads all sequences to the maximum length in the batch.

    Note: The model uses pad_id (0) as ignore_index for cross-entropy loss,
    so padded labels will be automatically ignored during loss computation.

    Args:
        batch: List of examples from dataset
        pad_id: Token ID for padding (default: 0 = <PAD>)

    Returns:
        Batched tensors with padding
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)

    # Initialize tensors with pad_id (model ignores pad_id in loss)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Fill in actual values
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def create_dataloaders(
    db_path: str | Path,
    tokenizer: Optional[AsciiTokenizer] = None,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    config: Optional[DataConfig] = None,
    pin_memory: bool = False,
    augment_train: bool = False,
    augment_prob: float = 0.5,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        db_path: Path to SQLite database
        tokenizer: Tokenizer instance (or None to use default)
        batch_size: Batch size for both loaders
        val_split: Fraction of data for validation
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducible split
        config: DataConfig for pipeline settings
        pin_memory: Pin memory for GPU transfer
        augment_train: If True, apply data augmentation to training data
        augment_prob: Probability of augmenting each training example

    Returns:
        Tuple of (train_loader, val_loader)
    """

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Create dataset
    tokenizer = tokenizer or get_tokenizer()
    config = config or DataConfig(db_path=str(db_path))

    full_dataset = AsciiArtDataset(db_path, tokenizer, config)

    # Compute split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Dataset too small for split: {total_size} total, "
            f"would give {train_size} train, {val_size} val"
        )

    # Create reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Optionally wrap training dataset with augmentation
    if config.augment and not augment_train:
        augment_train = True
        augment_prob = config.augment_prob

    if augment_train:
        # Create augmented wrapper for training
        # Note: We wrap the full dataset and filter via indices
        augmented_full = AugmentedAsciiArtDataset(
            full_dataset,
            augment_prob=augment_prob,
        )
        # Re-create the split with the augmented dataset
        generator = torch.Generator().manual_seed(seed)
        train_dataset, _ = torch.utils.data.random_split(
            augmented_full,
            [train_size, val_size],
            generator=generator,
        )

    # Get pad_id from tokenizer
    pad_id = tokenizer.pad_token_id

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def create_single_loader(
    db_path: str | Path,
    tokenizer: Optional[AsciiTokenizer] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    config: Optional[DataConfig] = None,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a single DataLoader (for inference/debugging).

    Args:
        db_path: Path to SQLite database
        tokenizer: Tokenizer instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        config: DataConfig for pipeline settings
        pin_memory: Pin memory for GPU transfer

    Returns:
        DataLoader instance
    """
    tokenizer = tokenizer or get_tokenizer()
    config = config or DataConfig(db_path=str(db_path))

    dataset: Dataset = AsciiArtDataset(db_path, tokenizer, config)
    if config.augment:
        dataset = AugmentedAsciiArtDataset(dataset, augment_prob=config.augment_prob)
    pad_id = tokenizer.pad_token_id

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    # Quick test with test database

    # Find project root
    project_root = Path(__file__).parent.parent.parent
    test_db = project_root / "data" / "ascii_art_test.db"

    if not test_db.exists():
        print(f"Test database not found at {test_db}")
        print("Creating minimal test...")
        # Just test tokenizer integration
        tokenizer = get_tokenizer()
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        print("Tokenizer test passed!")
    else:
        print(f"Testing with database: {test_db}")

        # Create tokenizer
        tokenizer = get_tokenizer()
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

        # Create config with relaxed constraints for test data
        config = DataConfig(
            db_path=str(test_db),
            block_size=2048,
            min_chars=1,  # Accept very small art for testing
            charset="ascii",
        )

        # Create dataset
        dataset = AsciiArtDataset(test_db, tokenizer, config)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            # Test single item
            print("\nTesting single item retrieval...")
            item = dataset[0]
            print(f"  input_ids shape: {item['input_ids'].shape}")
            print(f"  labels shape: {item['labels'].shape}")

            # Decode and show sample
            decoded = tokenizer.decode(item["input_ids"].tolist())
            print(f"  Sample (first 100 chars): {decoded[:100]!r}...")

            # Test collate function
            print("\nTesting collate function...")
            batch = [dataset[i] for i in range(min(3, len(dataset)))]
            collated = collate_fn(batch)
            print(f"  Batch input_ids shape: {collated['input_ids'].shape}")
            print(f"  Batch attention_mask shape: {collated['attention_mask'].shape}")

            # Test DataLoader creation
            if len(dataset) >= 2:
                print("\nTesting DataLoader creation...")
                try:
                    train_loader, val_loader = create_dataloaders(
                        test_db,
                        tokenizer,
                        batch_size=2,
                        val_split=0.3,
                        num_workers=0,  # Use 0 for testing
                        config=config,
                    )
                    print(f"  Train batches: {len(train_loader)}")
                    print(f"  Val batches: {len(val_loader)}")

                    # Get one batch
                    for batch in train_loader:
                        print(f"  Sample batch input_ids: {batch['input_ids'].shape}")
                        print(f"  Sample batch labels: {batch['labels'].shape}")
                        break
                except ValueError as e:
                    print(f"  Skipping DataLoader test (dataset too small): {e}")

            print("\nAll tests passed!")
        else:
            print("No valid art found in database with current filters.")
            print("Try adjusting config.charset or config.min_chars")
