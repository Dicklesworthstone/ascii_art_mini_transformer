"""
Data augmentation strategies for ASCII art training.

Implements various augmentation techniques to improve model robustness:
- Whitespace padding variations
- Horizontal flip for symmetric patterns
- Character substitution for style variation
- Description paraphrasing
- Light noise injection

All augmentations are probabilistic and can be composed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, TypeVar, cast

T = TypeVar("T")


class RandomLike(Protocol):
    def random(self) -> float: ...

    def randint(self, a: int, b: int) -> int: ...

    def choice(self, seq: Sequence[T]) -> T: ...

    def sample(self, population: Sequence[T], k: int) -> list[T]: ...


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    # Probability of applying each augmentation
    padding_prob: float = 0.3
    char_substitution_prob: float = 0.2
    horizontal_flip_prob: float = 0.1
    description_paraphrase_prob: float = 0.3
    noise_prob: float = 0.05

    # Augmentation parameters
    max_padding_chars: int = 5
    noise_rate: float = 0.01  # Fraction of chars to corrupt
    min_art_width: int = 3  # Don't augment very small art


# Character substitution mappings for style variation
CHAR_SUBSTITUTIONS = {
    "*": ["+", "x", "X", "o", "O"],
    "-": ["=", "_", "~"],
    "|": ["!", "I", "l"],
    "/": ["\\", "|"],
    "\\": ["/", "|"],
    "+": ["*", "x", "#"],
    "#": ["*", "@", "%"],
    ".": [",", "'", "`"],
    ":": [";", "."],
    "o": ["0", "O", "@"],
    "O": ["0", "o", "@"],
    "=": ["-", "~"],
}

# Flip mappings for horizontal reflection
FLIP_CHARS = {
    "/": "\\",
    "\\": "/",
    "(": ")",
    ")": "(",
    "[": "]",
    "]": "[",
    "{": "}",
    "}": "{",
    "<": ">",
    ">": "<",
    "d": "b",
    "b": "d",
    "p": "q",
    "q": "p",
}

# Description templates for paraphrasing
DESCRIPTION_TEMPLATES = [
    "{noun}",
    "a {noun}",
    "an {noun}",
    "the {noun}",
    "{noun} art",
    "{noun} drawing",
    "{noun} picture",
    "ASCII {noun}",
    "text art of {noun}",
    "simple {noun}",
]


def augment_padding(
    art: str,
    config: AugmentationConfig,
    *,
    rng: RandomLike | None = None,
) -> str:
    """
    Add random whitespace padding to the art.

    This teaches the model that leading/trailing whitespace doesn't
    change the meaning of the art.

    Args:
        art: ASCII art string
        config: Augmentation configuration

    Returns:
        Art with random padding added
    """
    rng = cast(RandomLike, random) if rng is None else rng
    if rng.random() > config.padding_prob:
        return art

    lines = art.split("\n")
    if not lines:
        return art

    # Random padding amounts
    left_pad = rng.randint(0, config.max_padding_chars)
    top_pad = rng.randint(0, config.max_padding_chars // 2)
    bottom_pad = rng.randint(0, config.max_padding_chars // 2)

    # Add left padding to each line
    if left_pad > 0:
        lines = [" " * left_pad + line for line in lines]

    # Add top/bottom blank lines
    lines = [""] * top_pad + lines + [""] * bottom_pad

    return "\n".join(lines)


def augment_char_substitution(
    art: str,
    config: AugmentationConfig,
    *,
    rng: RandomLike | None = None,
) -> str:
    """
    Substitute some characters with visually similar alternatives.

    This teaches the model to be robust to different character choices
    for the same visual pattern.

    Args:
        art: ASCII art string
        config: Augmentation configuration

    Returns:
        Art with some characters substituted
    """
    rng = cast(RandomLike, random) if rng is None else rng
    if rng.random() > config.char_substitution_prob:
        return art

    # Pick a random subset of substitutable chars to replace
    chars_to_sub = [c for c in CHAR_SUBSTITUTIONS if c in art]
    if not chars_to_sub:
        return art

    # Pick 1-3 character types to substitute
    num_to_sub = min(len(chars_to_sub), rng.randint(1, 3))
    selected = rng.sample(chars_to_sub, num_to_sub)

    result = art
    for char in selected:
        replacement = rng.choice(CHAR_SUBSTITUTIONS[char])
        result = result.replace(char, replacement)

    return result


def can_flip_horizontally(art: str) -> bool:
    """
    Check if art is suitable for horizontal flipping.

    Only flip art that appears roughly symmetric or uses simple patterns.
    """
    lines = art.strip().split("\n")
    if not lines:
        return False

    # Check if lines are roughly similar width (suggests structured art)
    widths = [len(line.rstrip()) for line in lines if line.strip()]
    if not widths:
        return False

    max_width = max(widths)
    min_width = min(widths)

    # If width varies too much, probably not good for flipping
    if max_width - min_width > max_width * 0.5:
        return False

    # Count flippable characters
    flippable = sum(1 for c in art if c in FLIP_CHARS)
    total = sum(1 for c in art if c not in " \n\t")

    # Flip if at least 10% of chars are flippable
    return total > 0 and flippable / total > 0.1


def augment_horizontal_flip(
    art: str,
    config: AugmentationConfig,
    *,
    rng: RandomLike | None = None,
) -> str:
    """
    Flip art horizontally for data augmentation.

    Only applies to art that appears suitable for flipping.
    Characters like / become \\ and vice versa.

    Args:
        art: ASCII art string
        config: Augmentation configuration

    Returns:
        Horizontally flipped art (or original if not suitable)
    """
    rng = cast(RandomLike, random) if rng is None else rng
    if rng.random() > config.horizontal_flip_prob:
        return art

    if not can_flip_horizontally(art):
        return art

    lines = art.split("\n")

    # Find max width for proper alignment
    max_width = max(len(line) for line in lines) if lines else 0
    if max_width < config.min_art_width:
        return art

    flipped_lines = []
    for line in lines:
        # Pad to max width, flip, then strip trailing spaces
        padded = line.ljust(max_width)
        flipped = ""
        for c in reversed(padded):
            flipped += FLIP_CHARS.get(c, c)
        flipped_lines.append(flipped.rstrip())

    return "\n".join(flipped_lines)


def augment_description(
    description: str,
    config: AugmentationConfig,
    *,
    rng: RandomLike | None = None,
) -> str:
    """
    Paraphrase the description for variety.

    Args:
        description: Original description
        config: Augmentation configuration

    Returns:
        Paraphrased description
    """
    rng = cast(RandomLike, random) if rng is None else rng
    if rng.random() > config.description_paraphrase_prob:
        return description

    # Extract the main noun/concept from the description
    # Simple heuristic: take the last significant word
    words = description.lower().strip().split()
    if not words:
        return description

    # Remove common articles and prefixes
    skip_words = {"a", "an", "the", "ascii", "art", "text", "simple", "drawing"}
    nouns = [w for w in words if w not in skip_words and len(w) > 2]

    if not nouns:
        return description

    noun = nouns[-1]  # Take the last content word

    # Apply a random template
    template = rng.choice(DESCRIPTION_TEMPLATES)

    # Handle a/an
    if "{noun}" in template:
        result = template.format(noun=noun)
        # Fix a/an if needed
        if result.startswith("a ") and noun[0] in "aeiou":
            result = "an " + noun
        elif result.startswith("an ") and noun[0] not in "aeiou":
            result = "a " + noun
        return result

    return description


def augment_noise(
    art: str,
    config: AugmentationConfig,
    *,
    rng: RandomLike | None = None,
) -> str:
    """
    Add very light noise to the art.

    Randomly corrupts a small fraction of non-whitespace characters.
    This teaches robustness to OCR-like errors.

    Args:
        art: ASCII art string
        config: Augmentation configuration

    Returns:
        Art with light noise added
    """
    rng = cast(RandomLike, random) if rng is None else rng
    if rng.random() > config.noise_prob:
        return art

    chars = list(art)
    num_to_corrupt = int(len(chars) * config.noise_rate)

    if num_to_corrupt < 1:
        return art

    # Find indices of non-whitespace, non-newline characters
    candidates = [i for i, c in enumerate(chars) if c not in " \n\t"]

    if not candidates:
        return art

    # Corrupt random characters
    to_corrupt = rng.sample(candidates, min(num_to_corrupt, len(candidates)))
    noise_chars = list("*#@+-.,:;'\"")

    for idx in to_corrupt:
        chars[idx] = rng.choice(noise_chars)

    return "".join(chars)


def augment_art(
    art: str,
    description: str,
    config: Optional[AugmentationConfig] = None,
    *,
    rng: RandomLike | None = None,
) -> tuple[str, str]:
    """
    Apply all augmentations to an art piece.

    Args:
        art: ASCII art string
        description: Art description
        config: Augmentation configuration (uses defaults if None)

    Returns:
        Tuple of (augmented_art, augmented_description)
    """
    if config is None:
        config = AugmentationConfig()

    rng = cast(RandomLike, random) if rng is None else rng

    # Apply augmentations in order
    # Note: order matters - some augmentations interact
    art = augment_horizontal_flip(art, config, rng=rng)  # Do flip before substitution
    art = augment_char_substitution(art, config, rng=rng)
    art = augment_padding(art, config, rng=rng)
    art = augment_noise(art, config, rng=rng)  # Noise last

    description = augment_description(description, config, rng=rng)

    return art, description


def validate_augmented_art(original: str, augmented: str) -> bool:
    """
    Validate that augmented art is still valid.

    Checks:
    - Non-empty
    - Has visual content (not just whitespace)
    - Reasonable size (not too small or too large)

    Args:
        original: Original art
        augmented: Augmented art

    Returns:
        True if augmented art is valid
    """
    if not augmented or not augmented.strip():
        return False

    # Must have some non-whitespace content
    content_chars = sum(1 for c in augmented if c not in " \n\t")
    if content_chars < 3:
        return False

    # Size shouldn't have changed too dramatically
    orig_lines = original.strip().split("\n")
    aug_lines = augmented.strip().split("\n")

    # Allow up to 2x size change
    if len(aug_lines) > len(orig_lines) * 2 + 10:
        return False

    return True


if __name__ == "__main__":
    # Test augmentations
    test_art = r"""
   /\_/\
  ( o.o )
   > ^ <
  /|   |\
 (_|   |_)
""".strip()

    print("Original art:")
    print(test_art)
    print()

    config = AugmentationConfig(
        padding_prob=1.0,
        char_substitution_prob=1.0,
        horizontal_flip_prob=0.0,  # Skip flip for cat (asymmetric)
        description_paraphrase_prob=1.0,
        noise_prob=0.0,  # Skip noise for readability
    )

    augmented_art, augmented_desc = augment_art(test_art, "a cute cat", config)

    print("Augmented art:")
    print(augmented_art)
    print()
    print("Original desc: a cute cat")
    print(f"Augmented desc: {augmented_desc}")
    print()

    # Test horizontal flip on symmetric art
    symmetric_art = r"""
  /\
 /  \
/____\
""".strip()

    print("Symmetric art (original):")
    print(symmetric_art)
    print()

    flip_config = AugmentationConfig(horizontal_flip_prob=1.0)
    flipped, _ = augment_art(symmetric_art, "triangle", flip_config)
    print("Flipped:")
    print(flipped)
    print()

    # Validation test
    print("Validation tests:")
    print(f"  Original valid: {validate_augmented_art(test_art, test_art)}")
    print(f"  Augmented valid: {validate_augmented_art(test_art, augmented_art)}")
    print(f"  Empty invalid: {validate_augmented_art(test_art, '')}")
    print(f"  Whitespace invalid: {validate_augmented_art(test_art, '   ')}")
