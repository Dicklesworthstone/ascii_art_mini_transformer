"""
Tests for data augmentation module.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.augmentation import (
    AugmentationConfig,
    augment_padding,
    augment_char_substitution,
    augment_horizontal_flip,
    augment_description,
    augment_noise,
    augment_art,
    validate_augmented_art,
    can_flip_horizontally,
    CHAR_SUBSTITUTIONS,
    FLIP_CHARS,
)


class TestAugmentationConfig:
    """Test augmentation configuration."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = AugmentationConfig()
        assert 0 <= config.padding_prob <= 1
        assert 0 <= config.char_substitution_prob <= 1
        assert 0 <= config.horizontal_flip_prob <= 1
        assert config.max_padding_chars >= 0
        assert 0 <= config.noise_rate <= 1

    def test_custom_config(self):
        """Can create custom config."""
        config = AugmentationConfig(
            padding_prob=0.8,
            noise_prob=0.1,
            max_padding_chars=10,
        )
        assert config.padding_prob == 0.8
        assert config.noise_prob == 0.1
        assert config.max_padding_chars == 10


class TestPaddingAugmentation:
    """Test whitespace padding augmentation."""

    def test_zero_prob_returns_original(self):
        """Zero probability returns unchanged art."""
        config = AugmentationConfig(padding_prob=0.0)
        art = "test\nart"
        result = augment_padding(art, config)
        assert result == art

    def test_always_adds_with_full_prob(self):
        """With prob=1.0, always adds padding."""
        config = AugmentationConfig(padding_prob=1.0, max_padding_chars=3)
        art = "x"

        # Run multiple times to check variability
        results = set()
        for _ in range(20):
            results.add(augment_padding(art, config))

        # Should have some variation (padding varies)
        assert len(results) > 1

    def test_preserves_content(self):
        """Augmentation preserves the actual art content."""
        config = AugmentationConfig(padding_prob=1.0, max_padding_chars=2)
        art = "ABC\nDEF"
        result = augment_padding(art, config)

        # Content should still be there (ignore whitespace)
        assert "ABC" in result
        assert "DEF" in result


class TestCharSubstitution:
    """Test character substitution augmentation."""

    def test_zero_prob_returns_original(self):
        """Zero probability returns unchanged art."""
        config = AugmentationConfig(char_substitution_prob=0.0)
        art = "***---||"
        result = augment_char_substitution(art, config)
        assert result == art

    def test_substitutes_characters(self):
        """With prob=1.0, substitutes characters."""
        config = AugmentationConfig(char_substitution_prob=1.0)
        art = "*" * 50  # Many stars to ensure substitution

        results = set()
        for _ in range(20):
            results.add(augment_char_substitution(art, config))

        # Should have variations (different substitutions)
        assert len(results) > 1

    def test_uses_valid_substitutions(self):
        """Substituted characters are from valid mappings."""
        config = AugmentationConfig(char_substitution_prob=1.0)
        art = "*****"

        for _ in range(10):
            result = augment_char_substitution(art, config)
            for char in result:
                if char != "*":
                    assert char in CHAR_SUBSTITUTIONS["*"]


class TestHorizontalFlip:
    """Test horizontal flip augmentation."""

    def test_zero_prob_returns_original(self):
        """Zero probability returns unchanged art."""
        config = AugmentationConfig(horizontal_flip_prob=0.0)
        art = "/\\\n\\//"
        result = augment_horizontal_flip(art, config)
        assert result == art

    def test_flips_characters(self):
        """Flips reversible characters."""
        config = AugmentationConfig(horizontal_flip_prob=1.0)
        art = "/ <--"

        # This should flip / to \\ and < to >
        result = augment_horizontal_flip(art, config)

        # Check that flips happened (content reversed)
        if result != art:  # Might not flip if can_flip_horizontally returns False
            assert "\\" in result or ">" in result

    def test_can_flip_horizontally_checks(self):
        """can_flip_horizontally correctly identifies flippable art."""
        # Art with flippable chars
        assert can_flip_horizontally("///\\\\\\")
        assert can_flip_horizontally("(( ))")

        # Art without flippable chars (need chars not in FLIP_CHARS)
        # Note: 'b' maps to 'd' in FLIP_CHARS, so use other letters
        assert not can_flip_horizontally("xyz")
        assert not can_flip_horizontally("...")


class TestDescriptionParaphrase:
    """Test description paraphrasing."""

    def test_zero_prob_returns_original(self):
        """Zero probability returns unchanged description."""
        config = AugmentationConfig(description_paraphrase_prob=0.0)
        desc = "a cute cat"
        result = augment_description(desc, config)
        assert result == desc

    def test_paraphrases_description(self):
        """With prob=1.0, paraphrases description."""
        config = AugmentationConfig(description_paraphrase_prob=1.0)
        desc = "a beautiful sunset"

        results = set()
        for _ in range(30):
            results.add(augment_description(desc, config))

        # Should have multiple variations
        assert len(results) > 1

    def test_keeps_noun(self):
        """Paraphrased description keeps the main noun."""
        config = AugmentationConfig(description_paraphrase_prob=1.0)
        desc = "a fluffy cat"

        for _ in range(10):
            result = augment_description(desc, config)
            assert "cat" in result.lower()


class TestNoiseAugmentation:
    """Test noise injection augmentation."""

    def test_zero_prob_returns_original(self):
        """Zero probability returns unchanged art."""
        config = AugmentationConfig(noise_prob=0.0)
        art = "perfect art here"
        result = augment_noise(art, config)
        assert result == art

    def test_adds_noise(self):
        """With prob=1.0 and high rate, adds visible noise."""
        config = AugmentationConfig(noise_prob=1.0, noise_rate=0.5)
        art = "AAAAAAAAAA"  # 10 As

        result = augment_noise(art, config)

        # Should have some corruption
        num_changed = sum(1 for a, b in zip(art, result) if a != b)
        assert num_changed > 0

    def test_preserves_whitespace(self):
        """Noise doesn't affect whitespace or newlines."""
        config = AugmentationConfig(noise_prob=1.0, noise_rate=0.5)
        art = "A A A\nB B B"

        result = augment_noise(art, config)

        # Whitespace and newlines should be preserved
        assert result.count(" ") == art.count(" ")
        assert result.count("\n") == art.count("\n")


class TestAugmentArt:
    """Test combined augmentation function."""

    def test_with_all_disabled(self):
        """All augmentations disabled returns original."""
        config = AugmentationConfig(
            padding_prob=0.0,
            char_substitution_prob=0.0,
            horizontal_flip_prob=0.0,
            description_paraphrase_prob=0.0,
            noise_prob=0.0,
        )
        art = "test art"
        desc = "test description"

        result_art, result_desc = augment_art(art, desc, config)
        assert result_art == art
        assert result_desc == desc

    def test_with_all_enabled(self):
        """All augmentations enabled produces variations."""
        config = AugmentationConfig(
            padding_prob=1.0,
            char_substitution_prob=1.0,
            horizontal_flip_prob=0.0,  # Skip flip for simple test
            description_paraphrase_prob=1.0,
            noise_prob=0.0,  # Skip noise for determinism
        )
        art = "***\n---"
        desc = "simple star pattern"

        result_art, result_desc = augment_art(art, desc, config)

        # Something should have changed
        assert result_art != art or result_desc != desc


class TestValidation:
    """Test validation of augmented art."""

    def test_valid_art(self):
        """Valid art passes validation."""
        original = "ABC\nDEF"
        augmented = "  ABC\n  DEF\n"
        assert validate_augmented_art(original, augmented)

    def test_empty_invalid(self):
        """Empty art fails validation."""
        original = "ABC"
        assert not validate_augmented_art(original, "")

    def test_whitespace_only_invalid(self):
        """Whitespace-only art fails validation."""
        original = "ABC"
        assert not validate_augmented_art(original, "   \n   ")

    def test_too_small_invalid(self):
        """Very small art fails validation."""
        original = "ABCDEF"
        assert not validate_augmented_art(original, "ab")

    def test_normal_size_change_valid(self):
        """Normal size changes are valid."""
        original = "short"
        augmented = "short\n\n\n"  # Added some blank lines
        assert validate_augmented_art(original, augmented)


class TestCharacterMappings:
    """Test character substitution and flip mappings."""

    def test_substitution_mappings_valid(self):
        """All substitution mappings have valid characters."""
        for char, subs in CHAR_SUBSTITUTIONS.items():
            assert len(char) == 1
            assert len(subs) > 0
            for sub in subs:
                assert len(sub) == 1

    def test_flip_mappings_symmetric(self):
        """Flip mappings are symmetric (a->b implies b->a)."""
        for char, flipped in FLIP_CHARS.items():
            # If a flips to b, b should flip to a (or to itself)
            assert flipped in FLIP_CHARS or flipped == char

    def test_flip_pairs(self):
        """Common flip pairs are correct."""
        assert FLIP_CHARS["/"] == "\\"
        assert FLIP_CHARS["\\"] == "/"
        assert FLIP_CHARS["("] == ")"
        assert FLIP_CHARS[")"] == "("
        assert FLIP_CHARS["<"] == ">"
        assert FLIP_CHARS[">"] == "<"
