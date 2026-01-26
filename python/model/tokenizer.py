"""
Character-level tokenizer for ASCII art generation.

This tokenizer preserves exact character positions, which is critical for
ASCII art where spatial structure matters. Unlike BPE tokenizers, this
ensures consistent token boundaries and a small vocabulary (~107 tokens).

Features:
- Character-level encoding (no subword splitting)
- Special tokens for BOS/EOS/PAD/UNK
- Constraint tokens for width/height
- Style tokens for different art types
- Explicit newline token for 2D position tracking
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


# Special token definitions with fixed IDs
SPECIAL_TOKENS = {
    "<PAD>": 0,  # Padding token
    "<BOS>": 1,  # Begin of sequence
    "<EOS>": 2,  # End of sequence
    "<UNK>": 3,  # Unknown character fallback
    "<SEP>": 4,  # Separator (description | art)
    "<WIDTH>": 5,  # Width constraint marker
    "<HEIGHT>": 6,  # Height constraint marker
    "<NEWLINE>": 7,  # Explicit newline for 2D tracking
    "<STYLE_ART>": 8,  # Realistic ASCII art style
    "<STYLE_BANNER>": 9,  # FIGlet-style text banner
    "<STYLE_SIMPLE>": 10,  # Simple line drawing
    "<STYLE_DETAILED>": 11,  # Detailed with shading
}

# Reverse mapping for decoding
SPECIAL_TOKEN_IDS = {v: k for k, v in SPECIAL_TOKENS.items()}

# Style token name to ID mapping
STYLE_TOKENS = {
    "art": "<STYLE_ART>",
    "banner": "<STYLE_BANNER>",
    "simple": "<STYLE_SIMPLE>",
    "detailed": "<STYLE_DETAILED>",
}

# Number of special tokens (used as offset for printable chars)
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

# Printable ASCII characters (space through tilde)
PRINTABLE_ASCII = [chr(i) for i in range(32, 127)]  # 95 characters


@dataclass
class TokenizerConfig:
    """Configuration for the ASCII art tokenizer."""

    vocab_size: int
    num_special_tokens: int
    printable_offset: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int
    sep_token_id: int
    newline_token_id: int


class AsciiTokenizer:
    """
    Character-level tokenizer for ASCII art.

    Training format:
        <BOS><WIDTH>40<HEIGHT>20<STYLE_ART>a cute cat sitting<SEP>
         /\\_/\\
        ( o.o )
         > ^ <
        <EOS>

    Each character maps to a unique token ID. Special tokens have fixed IDs
    at the start of the vocabulary.
    """

    def __init__(self) -> None:
        """Initialize the tokenizer with vocabulary mappings."""
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build the vocabulary mapping."""
        # Add special tokens first (IDs 0-11)
        for token_name, token_id in SPECIAL_TOKENS.items():
            self._char_to_id[token_name] = token_id
            self._id_to_char[token_id] = token_name

        # Add printable ASCII characters (IDs 12-106)
        for i, char in enumerate(PRINTABLE_ASCII):
            token_id = NUM_SPECIAL_TOKENS + i
            self._char_to_id[char] = token_id
            self._id_to_char[token_id] = char

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return len(self._char_to_id)

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return SPECIAL_TOKENS["<PAD>"]

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return SPECIAL_TOKENS["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return SPECIAL_TOKENS["<EOS>"]

    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return SPECIAL_TOKENS["<UNK>"]

    @property
    def sep_token_id(self) -> int:
        """Separator token ID."""
        return SPECIAL_TOKENS["<SEP>"]

    @property
    def newline_token_id(self) -> int:
        """Newline token ID."""
        return SPECIAL_TOKENS["<NEWLINE>"]

    def get_config(self) -> TokenizerConfig:
        """Get tokenizer configuration for model initialization."""
        return TokenizerConfig(
            vocab_size=self.vocab_size,
            num_special_tokens=NUM_SPECIAL_TOKENS,
            printable_offset=NUM_SPECIAL_TOKENS,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            unk_token_id=self.unk_token_id,
            sep_token_id=self.sep_token_id,
            newline_token_id=self.newline_token_id,
        )

    def _encode_number(self, n: int) -> list[int]:
        """Encode a number as character tokens (digit by digit)."""
        return [self._char_to_id[c] for c in str(n)]

    def _encode_text(self, text: str) -> list[int]:
        """Encode text, replacing unknown chars with <UNK>."""
        tokens = []
        for char in text:
            if char in self._char_to_id:
                tokens.append(self._char_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        return tokens

    def _encode_art(self, art: str) -> list[int]:
        """Encode ASCII art, converting newlines to <NEWLINE> token."""
        tokens = []
        for char in art:
            if char == "\n":
                tokens.append(self.newline_token_id)
            elif char in self._char_to_id:
                tokens.append(self._char_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Encode plain text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: If True, wrap with <BOS> and <EOS>

        Returns:
            List of token IDs
        """
        tokens = self._encode_text(text)
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def encode_training_example(
        self,
        description: str,
        art: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        style: str = "art",
    ) -> list[int]:
        """
        Encode a complete training example with constraints.

        Format: <BOS>[constraints][style][description]<SEP>[art]<EOS>

        Args:
            description: Text description of the art
            art: The ASCII art content
            width: Optional width constraint
            height: Optional height constraint
            style: Style type ('art', 'banner', 'simple', 'detailed')

        Returns:
            List of token IDs
        """
        tokens = [self.bos_token_id]

        # Add width constraint if provided
        if width is not None:
            tokens.append(self._char_to_id["<WIDTH>"])
            tokens.extend(self._encode_number(width))

        # Add height constraint if provided
        if height is not None:
            tokens.append(self._char_to_id["<HEIGHT>"])
            tokens.extend(self._encode_number(height))

        # Add style token (single token, not encoded string)
        style_token = STYLE_TOKENS.get(style, "<STYLE_ART>")
        tokens.append(self._char_to_id[style_token])

        # Add description
        tokens.extend(self._encode_text(description))

        # Add separator
        tokens.append(self.sep_token_id)

        # Add art with <NEWLINE> tokens
        tokens.extend(self._encode_art(art))

        # Add end token
        tokens.append(self.eos_token_id)

        return tokens

    def encode_inference_prompt(
        self,
        description: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        style: str = "art",
    ) -> list[int]:
        """
        Encode a prompt for inference (everything before the art).

        Format: <BOS>[constraints][style][description]<SEP>

        Args:
            description: Text description of desired art
            width: Optional width constraint
            height: Optional height constraint
            style: Style type ('art', 'banner', 'simple', 'detailed')

        Returns:
            List of token IDs ending with <SEP>
        """
        tokens = [self.bos_token_id]

        # Add constraints if provided
        if width is not None:
            tokens.append(self._char_to_id["<WIDTH>"])
            tokens.extend(self._encode_number(width))
        if height is not None:
            tokens.append(self._char_to_id["<HEIGHT>"])
            tokens.extend(self._encode_number(height))

        # Add style token
        style_token = STYLE_TOKENS.get(style, "<STYLE_ART>")
        tokens.append(self._char_to_id[style_token])

        # Add description
        tokens.extend(self._encode_text(description))

        # Add separator (model generates art after this)
        tokens.append(self.sep_token_id)

        return tokens

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: If True, omit special tokens from output

        Returns:
            Decoded string
        """
        chars = []
        for token_id in token_ids:
            if token_id not in self._id_to_char:
                if not skip_special_tokens:
                    chars.append(f"<UNK:{token_id}>")
                continue

            char = self._id_to_char[token_id]

            # Handle special tokens
            if char.startswith("<") and char.endswith(">"):
                if char == "<NEWLINE>":
                    chars.append("\n")
                elif not skip_special_tokens:
                    chars.append(char)
            else:
                chars.append(char)

        return "".join(chars)

    def decode_art(self, token_ids: Sequence[int]) -> str:
        """
        Decode only the art portion (after <SEP>).

        Finds <SEP> token and decodes everything after it until <EOS>.
        """
        sep_id = self.sep_token_id
        eos_id = self.eos_token_id

        # Find separator
        try:
            sep_idx = list(token_ids).index(sep_id)
        except ValueError:
            # No separator found, decode everything
            return self.decode(token_ids, skip_special_tokens=True)

        # Find end token after separator
        art_tokens = []
        for token_id in token_ids[sep_idx + 1 :]:
            if token_id == eos_id:
                break
            art_tokens.append(token_id)

        return self.decode(art_tokens, skip_special_tokens=True)

    def get_special_token_id(self, name: str) -> int:
        """Get ID for a special token by name."""
        if name not in SPECIAL_TOKENS:
            raise ValueError(f"Unknown special token: {name}")
        return SPECIAL_TOKENS[name]

    def is_special_token(self, token_id: int) -> bool:
        """Check if a token ID is a special token."""
        return token_id < NUM_SPECIAL_TOKENS

    def save(self, path: str | Path) -> None:
        """
        Save tokenizer config to JSON for Rust compatibility.

        Args:
            path: Path to save JSON file
        """
        config = {
            "vocab_size": self.vocab_size,
            "num_special_tokens": NUM_SPECIAL_TOKENS,
            "printable_offset": NUM_SPECIAL_TOKENS,
            "special_tokens": SPECIAL_TOKENS,
            "style_tokens": STYLE_TOKENS,
            "char_to_id": {
                k: v
                for k, v in self._char_to_id.items()
                if k not in SPECIAL_TOKENS  # Only exclude special tokens, keep '<' char
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "AsciiTokenizer":
        """
        Load tokenizer from JSON config.

        Args:
            path: Path to JSON config file

        Returns:
            Loaded tokenizer instance
        """
        # For now, just return a fresh instance since vocab is fixed
        # In the future, could validate against saved config
        return cls()


# Convenience function for getting a tokenizer instance
def get_tokenizer() -> AsciiTokenizer:
    """Get a tokenizer instance."""
    return AsciiTokenizer()


if __name__ == "__main__":  # pragma: no cover
    # Quick test
    tokenizer = AsciiTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Config: {tokenizer.get_config()}")

    # Test encoding/decoding
    test_text = "Hello, World!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nText: {test_text!r}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded!r}")
    assert decoded == test_text, "Round-trip failed!"

    # Test training example
    art = " /\\_/\\\n( o.o )\n > ^ <"
    tokens = tokenizer.encode_training_example(
        description="a cute cat",
        art=art,
        width=10,
        height=3,
        style="art",
    )
    print(f"\nTraining example tokens: {tokens[:20]}... ({len(tokens)} total)")

    # Decode art portion
    decoded_art = tokenizer.decode_art(tokens)
    print(f"Decoded art:\n{decoded_art}")

    # Test inference prompt
    prompt_tokens = tokenizer.encode_inference_prompt(
        description="a snake",
        width=40,
        style="simple",
    )
    print(f"\nInference prompt tokens: {prompt_tokens}")
    print(f"Prompt: {tokenizer.decode(prompt_tokens, skip_special_tokens=False)}")

    print("\nAll tests passed!")
