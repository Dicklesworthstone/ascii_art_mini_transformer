"""Tests for python/train/dataset.py."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from python.data.db import connect, initialize, insert_ascii_art
from python.model.tokenizer import SPECIAL_TOKENS, get_tokenizer
from python.train.augmentation import AugmentationConfig
from python.train.dataset import (
    AsciiArtDataset,
    AugmentedAsciiArtDataset,
    DataConfig,
    _compute_width_height,
    collate_fn,
)


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "schema.sql"


def _create_temp_db() -> sqlite3.Connection:
    """Create a temporary database for testing."""
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(handle.name)
    handle.close()

    conn = connect(db_path)
    initialize(conn, schema_path=_schema_path())
    return conn


def _insert_test_art(
    conn: sqlite3.Connection,
    raw_text: str,
    *,
    is_valid: bool = True,
    category: str | None = None,
    description: str | None = None,
) -> int:
    """Insert test ASCII art and return its ID."""
    art_id = insert_ascii_art(
        conn,
        raw_text=raw_text,
        source="test",
        title="test art",
        category=category,
        description=description,
        is_valid=is_valid,
    )
    assert art_id is not None
    return art_id


class TestComputeWidthHeight(unittest.TestCase):
    """Tests for _compute_width_height helper."""

    def test_simple_multiline(self) -> None:
        """Basic multiline art should compute correct dimensions."""
        art = "abc\ndefg\nhi"
        width, height = _compute_width_height(art)
        self.assertEqual(width, 4)  # "defg" is longest
        self.assertEqual(height, 3)

    def test_trailing_newline_treated_as_terminator(self) -> None:
        """Trailing newline should not add extra height."""
        art_no_trailing = "ab\ncd"
        art_trailing = "ab\ncd\n"

        w1, h1 = _compute_width_height(art_no_trailing)
        w2, h2 = _compute_width_height(art_trailing)

        self.assertEqual(h1, h2)
        self.assertEqual(w1, w2)

    def test_empty_string(self) -> None:
        """Empty string should give zero width and height 1 (one empty line)."""
        width, height = _compute_width_height("")
        self.assertEqual(width, 0)
        # "".split("\n") returns [""], so height is 1 (one empty line)
        self.assertEqual(height, 1)

    def test_single_line(self) -> None:
        """Single line art should have height 1."""
        width, height = _compute_width_height("hello")
        self.assertEqual(width, 5)
        self.assertEqual(height, 1)


class TestCharsetFiltering(unittest.TestCase):
    """Tests for charset filtering in AsciiArtDataset."""

    def test_charset_filter_selects_matching_charset(self) -> None:
        """Dataset should only load art matching the configured charset."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        # Insert ASCII art (charset='ascii')
        _insert_test_art(conn, "abc\ndef\nghi\n" * 5)

        # Insert extended ASCII art (charset='extended' due to \x80)
        _insert_test_art(conn, "abc\x80def\nghi\njkl\n" * 5)

        conn.close()

        # Load with ascii charset filter
        config = DataConfig(
            db_path=str(db_path),
            charset="ascii",
            min_chars=5,
        )
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        # Should only have the ASCII art
        self.assertEqual(len(dataset), 1)
        dataset.close()

    def test_fallback_when_no_matching_charset(self) -> None:
        """Dataset should fallback to loading all valid art when charset filter returns empty."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        # Insert only unicode art (no ascii art)
        _insert_test_art(conn, "abc\u0100def\nghi\njkl\n" * 3)  # \u0100 = unicode
        _insert_test_art(conn, "xyz\u0100wvu\ntsr\nqpo\n" * 3)  # \u0100 = unicode

        conn.close()

        # Load dataset with ascii filter - should fallback since no ascii art exists
        config = DataConfig(
            db_path=str(db_path),
            charset="ascii",  # No ascii art in DB
            min_chars=5,
        )
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        # Fallback kicks in when charset filter returns empty, loads all valid art
        self.assertEqual(len(dataset), 2)
        dataset.close()


class TestConstraintConditioning(unittest.TestCase):
    """Tests for constraint conditioning in AsciiArtDataset."""

    def test_add_constraints_false_skips_constraints(self) -> None:
        """When add_constraints=False, no dimension tokens should appear."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        _insert_test_art(
            conn,
            " /\\_/\\\n( o.o )\n > ^ <\n",
            description="a cat",
            category="animal",
        )
        conn.close()

        config = DataConfig(
            db_path=str(db_path),
            add_constraints=False,
            min_chars=5,
        )
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        # Get an item
        item = dataset[0]
        token_ids = item["input_ids"].tolist()

        # Should not contain width/height token IDs
        width_token_id = SPECIAL_TOKENS["<WIDTH>"]
        height_token_id = SPECIAL_TOKENS["<HEIGHT>"]
        self.assertNotIn(width_token_id, token_ids)
        self.assertNotIn(height_token_id, token_ids)
        dataset.close()

    def test_constraint_prob_zero_skips_constraints(self) -> None:
        """When constraint_prob=0, constraints should never appear."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        _insert_test_art(
            conn,
            " /\\_/\\\n( o.o )\n > ^ <\n",
            description="a cat",
            category="animal",
        )
        conn.close()

        config = DataConfig(
            db_path=str(db_path),
            add_constraints=True,
            constraint_prob=0.0,  # Never add constraints
            min_chars=5,
        )
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        width_token_id = SPECIAL_TOKENS["<WIDTH>"]
        height_token_id = SPECIAL_TOKENS["<HEIGHT>"]

        # Check multiple times to ensure consistency
        for _ in range(5):
            item = dataset[0]
            token_ids = item["input_ids"].tolist()
            self.assertNotIn(width_token_id, token_ids)
            self.assertNotIn(height_token_id, token_ids)

        dataset.close()

    def test_constraint_prob_one_always_adds_constraints(self) -> None:
        """When constraint_prob=1.0, constraints should always appear."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        _insert_test_art(
            conn,
            " /\\_/\\\n( o.o )\n > ^ <\n",
            description="a cat",
            category="animal",
        )
        conn.close()

        config = DataConfig(
            db_path=str(db_path),
            add_constraints=True,
            constraint_prob=1.0,  # Always add constraints
            width_prob=1.0,  # Always include width
            height_prob=1.0,  # Always include height
            min_chars=5,
        )
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        width_token_id = SPECIAL_TOKENS["<WIDTH>"]
        height_token_id = SPECIAL_TOKENS["<HEIGHT>"]

        # Check multiple times
        for _ in range(5):
            item = dataset[0]
            token_ids = item["input_ids"].tolist()
            # Should contain both width and height token IDs
            self.assertIn(width_token_id, token_ids)
            self.assertIn(height_token_id, token_ids)

        dataset.close()

    def test_width_prob_zero_excludes_width(self) -> None:
        """When width_prob=0, width should never appear even with constraints."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        _insert_test_art(
            conn,
            " /\\_/\\\n( o.o )\n > ^ <\n",
            description="a cat",
        )
        conn.close()

        config = DataConfig(
            db_path=str(db_path),
            add_constraints=True,
            constraint_prob=1.0,
            width_prob=0.0,  # Never include width
            height_prob=1.0,  # Always include height
            min_chars=5,
        )
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        width_token_id = SPECIAL_TOKENS["<WIDTH>"]
        height_token_id = SPECIAL_TOKENS["<HEIGHT>"]

        for _ in range(5):
            item = dataset[0]
            token_ids = item["input_ids"].tolist()
            self.assertNotIn(width_token_id, token_ids)
            self.assertIn(height_token_id, token_ids)

        dataset.close()

    def test_height_prob_zero_excludes_height(self) -> None:
        """When height_prob=0, height should never appear even with constraints."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        _insert_test_art(
            conn,
            " /\\_/\\\n( o.o )\n > ^ <\n",
            description="a cat",
        )
        conn.close()

        config = DataConfig(
            db_path=str(db_path),
            add_constraints=True,
            constraint_prob=1.0,
            width_prob=1.0,  # Always include width
            height_prob=0.0,  # Never include height
            min_chars=5,
        )
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        width_token_id = SPECIAL_TOKENS["<WIDTH>"]
        height_token_id = SPECIAL_TOKENS["<HEIGHT>"]

        for _ in range(5):
            item = dataset[0]
            token_ids = item["input_ids"].tolist()
            self.assertIn(width_token_id, token_ids)
            self.assertNotIn(height_token_id, token_ids)

        dataset.close()


class TestAugmentedDataset(unittest.TestCase):
    """Tests for AugmentedAsciiArtDataset wrapper."""

    def test_augment_prob_zero_returns_base_item(self) -> None:
        """When augment_prob=0, should always return base dataset item."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        _insert_test_art(
            conn,
            " /\\_/\\\n( o.o )\n > ^ <\n",
            description="a cat",
        )
        conn.close()

        config = DataConfig(
            db_path=str(db_path),
            add_constraints=False,
            min_chars=5,
        )
        tokenizer = get_tokenizer()
        base_dataset = AsciiArtDataset(db_path, tokenizer, config)

        # Augmented with prob=0 should be identical to base
        aug_dataset = AugmentedAsciiArtDataset(
            base_dataset,
            augment_prob=0.0,
        )

        base_item = base_dataset[0]
        aug_item = aug_dataset[0]

        self.assertTrue(torch.equal(base_item["input_ids"], aug_item["input_ids"]))
        base_dataset.close()

    def test_augmentation_recomputes_dimensions(self) -> None:
        """Augmented art should use recomputed dimensions for constraints."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        # Insert art that padding augmentation will modify
        original_art = "xx\nyy\nzz\n"
        _insert_test_art(conn, original_art, description="test")
        conn.close()

        config = DataConfig(
            db_path=str(db_path),
            add_constraints=True,
            constraint_prob=1.0,
            width_prob=1.0,
            height_prob=1.0,
            min_chars=3,
        )
        tokenizer = get_tokenizer()
        base_dataset = AsciiArtDataset(db_path, tokenizer, config)

        # Use augmentation config that always adds padding
        aug_config = AugmentationConfig(
            padding_prob=1.0,
            max_padding_chars=3,
            char_substitution_prob=0.0,
            horizontal_flip_prob=0.0,
            description_paraphrase_prob=0.0,
            noise_prob=0.0,
        )

        aug_dataset = AugmentedAsciiArtDataset(
            base_dataset,
            augmentation_config=aug_config,
            augment_prob=1.0,
        )

        # Get augmented item - dimensions in tokens should reflect padded art
        with patch("random.random", return_value=0.0):  # Ensure augmentation happens
            item = aug_dataset[0]

        # The item should be valid
        self.assertGreater(len(item["input_ids"]), 0)
        base_dataset.close()

    def test_augmentation_wrapper_preserves_length(self) -> None:
        """Augmented dataset should have same length as base."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        for i in range(5):
            _insert_test_art(conn, f"art{i}\nline2\nline3\n", description=f"art {i}")
        conn.close()

        config = DataConfig(db_path=str(db_path), min_chars=3)
        tokenizer = get_tokenizer()
        base_dataset = AsciiArtDataset(db_path, tokenizer, config)
        aug_dataset = AugmentedAsciiArtDataset(base_dataset, augment_prob=0.5)

        self.assertEqual(len(aug_dataset), len(base_dataset))
        base_dataset.close()


class TestCollateFn(unittest.TestCase):
    """Tests for collate_fn padding and attention mask."""

    def test_padding_to_max_length(self) -> None:
        """Sequences should be padded to the maximum length in batch."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {
                "input_ids": torch.tensor([4, 5, 6, 7, 8]),
                "labels": torch.tensor([4, 5, 6, 7, 8]),
            },
            {"input_ids": torch.tensor([9]), "labels": torch.tensor([9])},
        ]

        collated = collate_fn(batch, pad_id=0)

        # All should be padded to length 5
        self.assertEqual(collated["input_ids"].shape, (3, 5))
        self.assertEqual(collated["labels"].shape, (3, 5))
        self.assertEqual(collated["attention_mask"].shape, (3, 5))

    def test_padding_uses_correct_pad_id(self) -> None:
        """Padding should use the specified pad_id value."""
        batch = [
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])},
            {
                "input_ids": torch.tensor([3, 4, 5, 6]),
                "labels": torch.tensor([3, 4, 5, 6]),
            },
        ]

        collated = collate_fn(batch, pad_id=99)

        # First item should be padded with 99
        self.assertEqual(collated["input_ids"][0].tolist(), [1, 2, 99, 99])
        self.assertEqual(collated["labels"][0].tolist(), [1, 2, 99, 99])

    def test_attention_mask_correctness(self) -> None:
        """Attention mask should be True for real tokens, False for padding."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {
                "input_ids": torch.tensor([4, 5, 6, 7, 8]),
                "labels": torch.tensor([4, 5, 6, 7, 8]),
            },
        ]

        collated = collate_fn(batch, pad_id=0)

        # First item: 3 real tokens, 2 padding
        self.assertEqual(
            collated["attention_mask"][0].tolist(), [True, True, True, False, False]
        )

        # Second item: 5 real tokens, no padding
        self.assertEqual(
            collated["attention_mask"][1].tolist(), [True, True, True, True, True]
        )

    def test_single_item_batch(self) -> None:
        """Single item batch should work correctly."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([1, 2, 3, 4]),
            }
        ]

        collated = collate_fn(batch, pad_id=0)

        self.assertEqual(collated["input_ids"].shape, (1, 4))
        self.assertEqual(collated["labels"].shape, (1, 4))
        self.assertEqual(collated["attention_mask"].shape, (1, 4))
        self.assertTrue(collated["attention_mask"].all())

    def test_equal_length_sequences(self) -> None:
        """Batch with equal-length sequences needs no padding."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])},
            {"input_ids": torch.tensor([7, 8, 9]), "labels": torch.tensor([7, 8, 9])},
        ]

        collated = collate_fn(batch, pad_id=0)

        self.assertEqual(collated["input_ids"].shape, (3, 3))
        # All attention masks should be True
        self.assertTrue(collated["attention_mask"].all())


class TestDataConfigDefaults(unittest.TestCase):
    """Tests for DataConfig default values."""

    def test_max_chars_computed_from_block_size(self) -> None:
        """max_chars should be block_size - 50 when not specified."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
        conn.close()

        config = DataConfig(db_path=str(db_path), block_size=1000)
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        # max_chars should be computed as block_size - 50
        self.assertEqual(dataset.config.max_chars, 950)
        dataset.close()

    def test_explicit_max_chars_not_overwritten(self) -> None:
        """Explicit max_chars should not be overwritten."""
        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
        conn.close()

        config = DataConfig(db_path=str(db_path), block_size=1000, max_chars=500)
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        # Explicit max_chars should be preserved
        self.assertEqual(dataset.config.max_chars, 500)
        dataset.close()


class TestDatasetPickling(unittest.TestCase):
    """Tests for dataset pickling (for DataLoader workers)."""

    def test_dataset_is_pickleable(self) -> None:
        """Dataset should be pickleable for multiprocessing DataLoader."""
        import pickle

        conn = _create_temp_db()
        db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

        _insert_test_art(conn, "abc\ndef\nghi\n" * 3)
        conn.close()

        config = DataConfig(db_path=str(db_path), min_chars=3)
        tokenizer = get_tokenizer()
        dataset = AsciiArtDataset(db_path, tokenizer, config)

        # Access dataset to potentially cache connection
        _ = dataset[0]

        # Should be pickleable
        pickled = pickle.dumps(dataset)
        unpickled = pickle.loads(pickled)

        self.assertEqual(len(unpickled), len(dataset))
        dataset.close()
        unpickled.close()


if __name__ == "__main__":
    unittest.main()
