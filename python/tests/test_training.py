"""
Unit tests for the training data pipeline.

Tests the AsciiArtDataset, collate function, and DataLoader creation.
Training loop tests will be added when the training script is implemented.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# Add parent to path for imports before importing project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if torch is not installed
torch = pytest.importorskip("torch")

from model.tokenizer import AsciiTokenizer, get_tokenizer  # noqa: E402
from train.dataset import (  # noqa: E402
    AugmentationConfig,
    AugmentedAsciiArtDataset,
    AsciiArtDataset,
    DataConfig,
    augment_art,
    collate_fn,
    create_dataloaders,
    create_single_loader,
)


@pytest.fixture
def tokenizer() -> AsciiTokenizer:
    """Create a tokenizer for testing."""
    return get_tokenizer()


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with test data."""
    db_path = tmp_path / "test_ascii.db"
    conn = sqlite3.connect(db_path)

    # Create schema
    conn.executescript("""
        CREATE TABLE ascii_art (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            raw_text TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT,
            description TEXT,
            category TEXT,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            total_chars INTEGER NOT NULL,
            non_space_chars INTEGER NOT NULL,
            char_density REAL,
            charset TEXT NOT NULL DEFAULT 'ascii',
            char_histogram TEXT,
            is_valid INTEGER DEFAULT 1
        );
    """)

    # Insert test data
    test_arts = [
        {
            "content_hash": "hash1",
            "raw_text": "  /\\_/\\\n ( o.o )\n  > ^ <",
            "source": "test",
            "title": "Cat",
            "description": "A cute cat face",
            "category": "animal",
            "width": 9,
            "height": 3,
            "total_chars": 27,
            "non_space_chars": 15,
            "charset": "ascii",
        },
        {
            "content_hash": "hash2",
            "raw_text": " __|__\n(  o o )\n \\  ^ /\n  |||",
            "source": "test",
            "title": "Robot",
            "description": "A simple robot",
            "category": "object",
            "width": 9,
            "height": 4,
            "total_chars": 35,
            "non_space_chars": 18,
            "charset": "ascii",
        },
        {
            "content_hash": "hash3",
            "raw_text": "###\n# #\n###",
            "source": "test",
            "title": "Box",
            "description": "A simple box",
            "category": "simple",
            "width": 3,
            "height": 3,
            "total_chars": 11,
            "non_space_chars": 8,
            "charset": "ascii",
        },
        {
            "content_hash": "hash4",
            "raw_text": "TEST\n====\nBanner",
            "source": "figlet",
            "title": "Test Banner",
            "description": "FIGlet banner test",
            "category": "banner",
            "width": 6,
            "height": 3,
            "total_chars": 16,
            "non_space_chars": 14,
            "charset": "ascii",
        },
        {
            "content_hash": "hash5",
            "raw_text": "Hello\nWorld",
            "source": "test",
            "title": "Hello",
            "description": "Hello World text",
            "category": "text",
            "width": 5,
            "height": 2,
            "total_chars": 11,
            "non_space_chars": 10,
            "charset": "ascii",
        },
    ]

    for art in test_arts:
        conn.execute("""
            INSERT INTO ascii_art
            (content_hash, raw_text, source, title, description, category,
             width, height, total_chars, non_space_chars, charset)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            art["content_hash"],
            art["raw_text"],
            art["source"],
            art["title"],
            art["description"],
            art["category"],
            art["width"],
            art["height"],
            art["total_chars"],
            art["non_space_chars"],
            art["charset"],
        ))

    conn.commit()
    conn.close()
    return db_path


class TestAsciiArtDataset:
    """Tests for AsciiArtDataset class."""

    def test_dataset_loading(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that dataset loads from SQLite correctly."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)
        assert len(dataset.ids) > 0, "Dataset should have loaded some IDs"

    def test_dataset_length(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that dataset reports correct length."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)
        assert len(dataset) == 5, f"Expected 5 items, got {len(dataset)}"

    def test_dataset_item_format(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that items have expected structure."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        item = dataset[0]

        # Check all expected keys
        assert "input_ids" in item, "Item should have input_ids"
        assert "labels" in item, "Item should have labels"
        assert "row_pos" in item, "Item should have row_pos"
        assert "col_pos" in item, "Item should have col_pos"

        # Check types
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        assert isinstance(item["row_pos"], torch.Tensor)
        assert isinstance(item["col_pos"], torch.Tensor)

        # Check dtypes
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long
        assert item["row_pos"].dtype == torch.long
        assert item["col_pos"].dtype == torch.long


def test_training_raises_on_empty_train_loader(temp_db: Path, tmp_path: Path) -> None:
    """
    Regression test: avoid silent infinite loop when train DataLoader yields no batches.

    `create_dataloaders(..., drop_last=True)` can produce a 0-length train loader if the
    requested `batch_size` exceeds the available train split size.
    """
    config = TrainingConfig(
        db_path=str(temp_db),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
        dtype="float32",
        batch_size=64,
        gradient_accumulation_steps=1,
        max_iters=0,
        val_split=0.2,
        num_workers=0,
    )

    with pytest.raises(ValueError, match=r"0 batches"):
        run_training(config)

    def test_input_label_same(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that labels equal input_ids (model does internal shifting)."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        item = dataset[0]

        # Labels should be same as input_ids (model shifts internally)
        assert len(item["labels"]) == len(item["input_ids"])
        assert torch.equal(item["labels"], item["input_ids"])

    def test_positions_shape(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that position tensors have correct shape."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        item = dataset[0]

        # Positions should match input length
        assert item["row_pos"].shape == item["input_ids"].shape
        assert item["col_pos"].shape == item["input_ids"].shape

    def test_constraint_conditioning(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that constraint conditioning produces valid sequences."""
        # Force constraints to be added
        config = DataConfig(
            db_path=str(temp_db),
            min_chars=1,
            add_constraints=True,
            constraint_prob=1.0,
            width_prob=1.0,
            height_prob=1.0,
        )
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        # Get an item - should have width/height tokens
        item = dataset[0]
        decoded = tokenizer.decode(item["input_ids"].tolist(), skip_special_tokens=False)

        # Should contain <WIDTH> and <HEIGHT> tokens
        assert "<WIDTH>" in decoded, "Should have width constraint"
        assert "<HEIGHT>" in decoded, "Should have height constraint"

    def test_no_constraints(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that disabling constraints works."""
        config = DataConfig(
            db_path=str(temp_db),
            min_chars=1,
            add_constraints=False,
        )
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        # Get multiple items to be sure
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            decoded = tokenizer.decode(item["input_ids"].tolist(), skip_special_tokens=False)

            # Should not have constraint tokens
            assert "<WIDTH>" not in decoded, "Should not have width constraint"
            assert "<HEIGHT>" not in decoded, "Should not have height constraint"


class TestCollateFn:
    """Tests for the collate function."""

    def test_batch_shapes(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that batches have correct shapes."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        # Create a batch
        batch_items = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = collate_fn(batch_items)

        batch_size = len(batch_items)
        max_len = max(len(item["input_ids"]) for item in batch_items)

        assert batch["input_ids"].shape == (batch_size, max_len)
        assert batch["labels"].shape == (batch_size, max_len)
        assert batch["row_pos"].shape == (batch_size, max_len)
        assert batch["col_pos"].shape == (batch_size, max_len)
        assert batch["attention_mask"].shape == (batch_size, max_len)

    def test_padding(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that sequences are padded correctly."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        batch_items = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = collate_fn(batch_items, pad_id=0)

        # Check that shorter sequences are padded with 0s
        for i, item in enumerate(batch_items):
            seq_len = len(item["input_ids"])
            max_len = batch["input_ids"].shape[1]

            # After sequence should be padded
            if seq_len < max_len:
                padded_region = batch["input_ids"][i, seq_len:]
                assert (padded_region == 0).all(), "Padded region should be zeros"

    def test_attention_mask(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that attention mask is correct."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        batch_items = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = collate_fn(batch_items)

        # Check mask matches actual sequence lengths
        for i, item in enumerate(batch_items):
            seq_len = len(item["input_ids"])
            mask = batch["attention_mask"][i]

            # True for actual tokens
            assert mask[:seq_len].all(), "Mask should be True for real tokens"

            # False for padding
            if seq_len < len(mask):
                assert not mask[seq_len:].any(), "Mask should be False for padding"

    def test_padded_labels(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that padded labels use pad_id (model ignores in loss)."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        batch_items = [dataset[i] for i in range(min(3, len(dataset)))]
        pad_id = tokenizer.pad_token_id
        batch = collate_fn(batch_items, pad_id=pad_id)

        # Check that padded positions have pad_id (0)
        for i, item in enumerate(batch_items):
            seq_len = len(item["input_ids"])
            labels = batch["labels"][i]

            if seq_len < len(labels):
                padded_labels = labels[seq_len:]
                assert (padded_labels == pad_id).all(), "Padded labels should be pad_id"


class TestDataLoaderCreation:
    """Tests for DataLoader factory functions."""

    def test_create_dataloaders(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that train/val loaders are created correctly."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)

        train_loader, val_loader = create_dataloaders(
            temp_db,
            tokenizer,
            batch_size=2,
            val_split=0.4,  # 40% to ensure we have at least 1 val
            num_workers=0,
            config=config,
        )

        assert len(train_loader) > 0, "Should have train batches"
        # With 5 items and 40% val split = 2 val, 3 train
        # With batch_size=2, drop_last=True: train has 1 batch

    def test_train_val_split(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that split is reproducible with seed."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)

        # Create loaders twice with same seed
        train1, val1 = create_dataloaders(
            temp_db, tokenizer, batch_size=1, val_split=0.4,
            num_workers=0, seed=42, config=config,
        )
        train2, val2 = create_dataloaders(
            temp_db, tokenizer, batch_size=1, val_split=0.4,
            num_workers=0, seed=42, config=config,
        )

        assert len(train1) == len(train2), "Same seed should give same split"
        assert len(val1) == len(val2), "Same seed should give same split"

    def test_create_single_loader(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test single loader creation."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)

        loader = create_single_loader(
            temp_db,
            tokenizer,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            config=config,
        )

        assert len(loader) > 0, "Should have batches"

    def test_batch_iteration(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that we can iterate through batches."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)

        loader = create_single_loader(
            temp_db,
            tokenizer,
            batch_size=2,
            num_workers=0,
            config=config,
        )

        batch_count = 0
        for batch in loader:
            batch_count += 1
            assert "input_ids" in batch
            assert "labels" in batch
            assert "attention_mask" in batch

        assert batch_count > 0, "Should have iterated through batches"


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig(db_path="test.db")

        assert config.block_size == 2048
        assert config.add_constraints is True
        assert config.constraint_prob == 0.8
        assert config.charset == "ascii"
        assert config.min_chars == 10

    def test_max_chars_auto(self):
        """Test that max_chars is computed from block_size."""
        config = DataConfig(db_path="test.db", block_size=1000)

        # max_chars is set to None initially, computed in Dataset
        assert config.max_chars is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataConfig(
            db_path="test.db",
            block_size=512,
            add_constraints=False,
            charset="extended",
            min_chars=5,
            max_chars=400,
        )

        assert config.block_size == 512
        assert config.add_constraints is False
        assert config.charset == "extended"
        assert config.min_chars == 5
        assert config.max_chars == 400


class TestAugmentation:
    """Tests for training-time data augmentation."""

    def test_augment_art_horizontal_flip(self):
        cfg = AugmentationConfig(
            padding_prob=0.0,
            char_substitution_prob=0.0,
            horizontal_flip_prob=1.0,
            description_paraphrase_prob=0.0,
            noise_prob=0.0,
        )
        art, desc = augment_art("((/", "desc", cfg)
        assert desc == "desc"
        assert art == "\\))"

    def test_create_dataloaders_uses_augmented_dataset(self, temp_db: Path, tokenizer: AsciiTokenizer):
        config = DataConfig(db_path=str(temp_db), min_chars=1, augment=True, augment_prob=1.0)

        train_loader, val_loader = create_dataloaders(
            temp_db,
            tokenizer,
            batch_size=2,
            val_split=0.4,
            num_workers=0,
            config=config,
        )

        # create_dataloaders uses random_split, so the loader datasets are Subset wrappers.
        assert isinstance(train_loader.dataset.dataset, AugmentedAsciiArtDataset)
        assert isinstance(val_loader.dataset.dataset, AsciiArtDataset)


class TestPositionComputation:
    """Tests for 2D position computation in dataset."""

    def test_row_positions_increment(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that row positions increment after newlines."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        item = dataset[0]
        row_pos = item["row_pos"]
        input_ids = item["input_ids"]

        # Find newline positions
        newline_id = tokenizer.newline_token_id
        newline_positions = (input_ids == newline_id).nonzero(as_tuple=True)[0]

        if len(newline_positions) > 0:
            # Row should increment after newline
            first_newline = newline_positions[0].item()
            if first_newline + 1 < len(row_pos):
                assert row_pos[first_newline + 1] > row_pos[first_newline], \
                    "Row should increment after newline"

    def test_column_positions_reset(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that column positions reset after newlines."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        item = dataset[0]
        col_pos = item["col_pos"]
        input_ids = item["input_ids"]

        # Find newline positions
        newline_id = tokenizer.newline_token_id
        newline_positions = (input_ids == newline_id).nonzero(as_tuple=True)[0]

        if len(newline_positions) > 0:
            # Column should reset to 0 after newline
            first_newline = newline_positions[0].item()
            if first_newline + 1 < len(col_pos):
                assert col_pos[first_newline + 1] == 0, \
                    "Column should reset to 0 after newline"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_database(self, tmp_path: Path, tokenizer: AsciiTokenizer):
        """Test handling of empty database."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE ascii_art (
                id INTEGER PRIMARY KEY,
                content_hash TEXT,
                raw_text TEXT,
                source TEXT,
                width INTEGER,
                height INTEGER,
                total_chars INTEGER,
                non_space_chars INTEGER,
                charset TEXT,
                is_valid INTEGER
            );
        """)
        conn.close()

        config = DataConfig(db_path=str(db_path), min_chars=1)
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        assert len(dataset) == 0, "Empty database should give empty dataset"

    def test_invalid_index(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that invalid index raises appropriate error."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        with pytest.raises(IndexError):
            _ = dataset[1000]  # Index way out of bounds

    def test_single_item_dataset(self, tmp_path: Path, tokenizer: AsciiTokenizer):
        """Test dataset with single item."""
        db_path = tmp_path / "single.db"
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE ascii_art (
                id INTEGER PRIMARY KEY,
                content_hash TEXT,
                raw_text TEXT,
                source TEXT,
                description TEXT,
                category TEXT,
                width INTEGER,
                height INTEGER,
                total_chars INTEGER,
                non_space_chars INTEGER,
                charset TEXT,
                is_valid INTEGER
            );
            INSERT INTO ascii_art VALUES
            (1, 'hash', 'Test', 'test', 'Test', 'test', 4, 1, 4, 4, 'ascii', 1);
        """)
        conn.close()

        config = DataConfig(db_path=str(db_path), min_chars=1)
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        assert len(dataset) == 1
        item = dataset[0]
        assert len(item["input_ids"]) > 0


# ============================================================================
# Training Loop Tests (require train.py - bd-2sf)
# ============================================================================

from train.train import (  # noqa: E402
    TrainingConfig,
    train as run_training,
    get_lr,
    save_checkpoint,
    load_checkpoint,
    _get_torch_dtype,
)
from model.transformer import create_model, get_small_config  # noqa: E402


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.batch_size == 64
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 6e-4
        assert config.max_iters == 100000
        assert config.warmup_iters == 2000

    def test_effective_batch_size(self):
        """Test effective batch size computation."""
        config = TrainingConfig(batch_size=32, gradient_accumulation_steps=8)
        assert config.effective_batch_size == 256

    def test_model_config_creation(self):
        """Test creating model config from training config."""
        config = TrainingConfig(n_layer=4, n_head=4, n_embd=256)
        model_config = config.get_model_config()

        assert model_config.n_layer == 4
        assert model_config.n_head == 4
        assert model_config.n_embd == 256


class TestLearningRateSchedule:
    """Tests for learning rate scheduling."""

    def test_warmup_starts_at_zero(self):
        """LR should start at 0 during warmup."""
        config = TrainingConfig(
            learning_rate=6e-4,
            warmup_iters=100,
        )

        lr_0 = get_lr(0, config)
        assert lr_0 == 0.0, "LR should start at 0"

    def test_warmup_increases(self):
        """LR should increase during warmup."""
        config = TrainingConfig(
            learning_rate=6e-4,
            warmup_iters=100,
        )

        lr_0 = get_lr(0, config)
        lr_50 = get_lr(50, config)
        lr_100 = get_lr(100, config)

        assert lr_50 > lr_0, "LR should increase during warmup"
        assert lr_100 >= lr_50, "LR should continue increasing"

    def test_warmup_reaches_max(self):
        """LR should reach max at end of warmup."""
        config = TrainingConfig(
            learning_rate=6e-4,
            warmup_iters=100,
        )

        lr_at_warmup = get_lr(100, config)
        assert abs(lr_at_warmup - 6e-4) < 1e-8, "LR should reach max at warmup end"

    def test_cosine_decay(self):
        """LR should decay after warmup."""
        config = TrainingConfig(
            learning_rate=6e-4,
            min_lr=6e-5,
            warmup_iters=100,
            lr_decay_iters=1000,
        )

        lr_100 = get_lr(100, config)  # End of warmup
        lr_500 = get_lr(500, config)  # Mid decay
        lr_1000 = get_lr(1000, config)  # End of decay

        assert lr_500 < lr_100, "LR should decrease after warmup"
        assert lr_1000 < lr_500, "LR should continue decreasing"

    def test_min_lr_floor(self):
        """LR should not go below min_lr."""
        config = TrainingConfig(
            learning_rate=6e-4,
            min_lr=6e-5,
            warmup_iters=100,
            lr_decay_iters=1000,
        )

        lr_after_decay = get_lr(2000, config)  # Well after decay
        assert lr_after_decay == config.min_lr, "LR should stay at min after decay"


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(self, tmp_path):
        """Test that checkpoint saves correctly."""
        model_config = get_small_config()
        model = create_model(model_config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = TrainingConfig(checkpoint_dir=str(tmp_path))

        ckpt_path = tmp_path / "test_ckpt.pt"
        save_checkpoint(model, optimizer, iter_num=100, best_val_loss=0.5,
                        config=config, path=ckpt_path)

        assert ckpt_path.exists(), "Checkpoint file should exist"

    def test_load_checkpoint(self, tmp_path):
        """Test that checkpoint loads correctly."""
        model_config = get_small_config()
        model = create_model(model_config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = TrainingConfig(checkpoint_dir=str(tmp_path))

        # Save
        ckpt_path = tmp_path / "test_ckpt.pt"
        save_checkpoint(model, optimizer, iter_num=100, best_val_loss=0.5,
                        config=config, path=ckpt_path)

        # Create new model and optimizer
        model2 = create_model(model_config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

        # Load
        iter_num, best_val_loss = load_checkpoint(ckpt_path, model2, optimizer2)

        assert iter_num == 100, "Iter num should be restored"
        assert best_val_loss == 0.5, "Best val loss should be restored"

    def test_checkpoint_weights_match(self, tmp_path):
        """Test that loaded weights match saved weights."""
        model_config = get_small_config()
        model = create_model(model_config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = TrainingConfig(checkpoint_dir=str(tmp_path))

        # Get original weights
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}

        # Save
        ckpt_path = tmp_path / "test_ckpt.pt"
        save_checkpoint(model, optimizer, iter_num=100, best_val_loss=0.5,
                        config=config, path=ckpt_path)

        # Create new model and load
        model2 = create_model(model_config)
        load_checkpoint(ckpt_path, model2)

        # Check weights match
        for key in original_weights:
            assert torch.equal(original_weights[key], model2.state_dict()[key]), \
                f"Weights should match for {key}"


class TestTrainingStep:
    """Tests for training step components."""

    def test_single_forward_pass(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test single forward pass with loss computation."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        if len(dataset) == 0:
            pytest.skip("Empty dataset")

        model_config = get_small_config()
        model = create_model(model_config)

        item = dataset[0]
        input_ids = item["input_ids"].unsqueeze(0)
        labels = item["labels"].unsqueeze(0)

        logits, loss = model(input_ids, labels=labels)

        assert logits is not None, "Should produce logits"
        assert loss is not None, "Should compute loss"
        assert loss.item() > 0, "Loss should be positive"

    def test_backward_pass(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test backward pass produces gradients."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        if len(dataset) == 0:
            pytest.skip("Empty dataset")

        model_config = get_small_config()
        model = create_model(model_config)

        item = dataset[0]
        input_ids = item["input_ids"].unsqueeze(0)
        labels = item["labels"].unsqueeze(0)

        _, loss = model(input_ids, labels=labels)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient should exist for {name}"

    def test_optimizer_step(self, temp_db: Path, tokenizer: AsciiTokenizer):
        """Test that optimizer updates parameters."""
        config = DataConfig(db_path=str(temp_db), min_chars=1)
        dataset = AsciiArtDataset(temp_db, tokenizer, config)

        if len(dataset) == 0:
            pytest.skip("Empty dataset")

        model_config = get_small_config()
        model = create_model(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Get original weights
        original_weight = model.token_embedding.weight.clone()

        item = dataset[0]
        input_ids = item["input_ids"].unsqueeze(0)
        labels = item["labels"].unsqueeze(0)

        _, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.equal(original_weight, model.token_embedding.weight), \
            "Weights should update after optimizer step"


class TestDtypeConversion:
    """Tests for dtype conversion utility."""

    def test_float32(self):
        """Test float32 conversion."""
        assert _get_torch_dtype("float32") == torch.float32

    def test_float16(self):
        """Test float16 conversion."""
        assert _get_torch_dtype("float16") == torch.float16

    def test_bfloat16(self):
        """Test bfloat16 conversion."""
        assert _get_torch_dtype("bfloat16") == torch.bfloat16

    def test_invalid_dtype(self):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError):
            _get_torch_dtype("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
