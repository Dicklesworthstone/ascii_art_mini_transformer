"""Model module for ASCII art transformer architecture."""

from .tokenizer import (
    AsciiTokenizer,
    TokenizerConfig,
    get_tokenizer,
    SPECIAL_TOKENS,
    STYLE_TOKENS,
    NUM_SPECIAL_TOKENS,
)

__all__ = [
    # Tokenizer
    "AsciiTokenizer",
    "TokenizerConfig",
    "get_tokenizer",
    "SPECIAL_TOKENS",
    "STYLE_TOKENS",
    "NUM_SPECIAL_TOKENS",
]

# Positional encoding depends on torch; keep tokenizer imports usable even
# when torch isn't installed (common during early repo bootstrap).
try:
    from .positional_encoding import (
        LearnedPositionalEncoding2D,
        SinusoidalPositionalEncoding2D,
        PositionalEncoding2DModule,
        compute_2d_positions_vectorized,
        compute_2d_positions_simple,
        create_positional_encoding,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__ += [
        "LearnedPositionalEncoding2D",
        "SinusoidalPositionalEncoding2D",
        "PositionalEncoding2DModule",
        "compute_2d_positions_vectorized",
        "compute_2d_positions_simple",
        "create_positional_encoding",
    ]

# Transformer model also depends on torch
try:
    from .transformer import (
        AsciiGPT,
        AsciiGPTConfig,
        CausalSelfAttention,
        TransformerBlock,
        MLP,
        create_model,
        get_small_config,
        get_medium_config,
        get_large_config,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__ += [
        "AsciiGPT",
        "AsciiGPTConfig",
        "CausalSelfAttention",
        "TransformerBlock",
        "MLP",
        "create_model",
        "get_small_config",
        "get_medium_config",
        "get_large_config",
    ]
