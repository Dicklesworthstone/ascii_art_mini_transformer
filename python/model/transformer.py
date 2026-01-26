"""
ASCII Art Transformer Model based on nanoGPT architecture.

This is a decoder-only transformer specifically designed for ASCII art generation.
Key modifications from standard GPT:
1. Character-level tokenization (~107 tokens vs 50k+ for BPE)
2. 2D positional encoding (row/column instead of 1D position)
3. Optimized size for CPU inference (10-30M parameters)

Based on:
- nanoGPT by Karpathy (https://github.com/karpathy/nanoGPT)
- ViTARC insights on 2D positional encoding
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.nn import functional as F

from .positional_encoding import PositionalEncoding2DModule


@dataclass
class AsciiGPTConfig:
    """Configuration for the ASCII art GPT model."""

    # Model architecture
    block_size: int = 2048  # Maximum sequence length
    vocab_size: int = 107  # Character-level (12 special + 95 printable)
    n_layer: int = 6  # Number of transformer blocks
    n_head: int = 6  # Number of attention heads
    n_embd: int = 384  # Embedding dimension
    dropout: float = 0.1  # Dropout probability

    # 2D positional encoding
    max_rows: int = 100  # Maximum supported rows
    max_cols: int = 200  # Maximum supported columns
    newline_token_id: int = 7  # Token ID for newline

    # Weight initialization
    init_std: float = 0.02  # Standard deviation for weight init

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.n_embd // self.n_head

    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        # Token embeddings (shared with lm_head via weight tying)
        emb = self.vocab_size * self.n_embd

        # 2D positional embeddings (row + column, each using half of n_embd)
        row_dim = self.n_embd // 2
        col_dim = self.n_embd - row_dim
        pos = self.max_rows * row_dim + self.max_cols * col_dim

        # Per transformer block
        # - LayerNorm: 2 * n_embd * 2 (two layernorms, weight + bias each)
        # - Attention: 4 * n_embd * n_embd (qkv + out projections)
        # - MLP: n_embd * 4 * n_embd * 2 (up + down projections)
        per_block = (
            4 * self.n_embd  # LayerNorms
            + 4 * self.n_embd * self.n_embd  # Attention
            + 2 * self.n_embd * 4 * self.n_embd  # MLP
        )
        blocks = self.n_layer * per_block

        # Final layer norm
        final_ln = 2 * self.n_embd

        return emb + pos + blocks + final_ln


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses flash attention when available for efficiency.
    """

    def __init__(self, config: AsciiGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self._cached_causal_mask: torch.Tensor | None = None

        # Key, query, value projections for all heads combined
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Lazily build and cache a causal mask for fallback attention paths."""
        mask = self._cached_causal_mask
        if mask is None or mask.device != device or mask.size(-1) < seq_len:
            m = torch.tril(
                torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
            )
            mask = m.view(1, 1, seq_len, seq_len)
            self._cached_causal_mask = mask
        return mask[:, :, :seq_len, :seq_len]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
            attention_mask: Optional mask for padding (batch, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.size()

        # Calculate query, key, values
        qkv = cast(torch.Tensor, self.c_attn(x))
        q, k, v = qkv.chunk(3, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # Prefer PyTorch scaled-dot-product attention when available to avoid materializing
        # the full (T, T) attention matrix (important for memory at large block sizes).
        #
        # Note: our PyTorch version rejects providing both `attn_mask` and `is_causal=True`.
        # For the common right-padding case, callers can omit `attention_mask` entirely because
        # causal masking already prevents earlier tokens from attending to later padded positions.
        if hasattr(F, "scaled_dot_product_attention") and attention_mask is None:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = cast(torch.Tensor, self.c_proj(y))
            y = cast(torch.Tensor, self.resid_dropout(y))
            return y

        # Fallback: explicit attention matrix (older PyTorch).
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask
        att = att.masked_fill(~self._get_causal_mask(T, att.device), float("-inf"))

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(padding_mask == 0, float("-inf"))

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = cast(torch.Tensor, self.c_proj(y))
        y = cast(torch.Tensor, self.resid_dropout(y))
        return y


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.

    Standard transformer MLP with 4x hidden dimension expansion.
    """

    def __init__(self, config: AsciiGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm architecture.

    Structure: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """

    def __init__(self, config: AsciiGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class AsciiGPT(nn.Module):
    """
    ASCII Art GPT - A decoder-only transformer for ASCII art generation.

    Features:
    - Character-level tokenization (~107 tokens)
    - 2D positional encoding (row + column)
    - Causal attention for autoregressive generation
    - Weight tying between embeddings and LM head
    """

    def __init__(self, config: AsciiGPTConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # 2D positional encoding
        self.pos_encoding = PositionalEncoding2DModule(
            d_model=config.n_embd,
            max_rows=config.max_rows,
            max_cols=config.max_cols,
            newline_token_id=config.newline_token_id,
            learned=True,
        )

        # Embedding dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"AsciiGPT initialized with {n_params:,} parameters", file=sys.stderr)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            labels: Optional labels for computing loss (batch, seq_len)

        Returns:
            logits: Predicted logits of shape (batch, seq_len, vocab_size)
            loss: Optional cross-entropy loss if labels provided
        """
        B, T = input_ids.size()

        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Token embeddings
        tok_emb = self.token_embedding(input_ids)  # (B, T, n_embd)

        # 2D positional embeddings
        pos_emb = self.pos_encoding(input_ids)  # (B, T, n_embd)

        # Combine embeddings
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Compute logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with this probability
            eos_token_id: If set, stop generation when this token is produced

        Returns:
            Generated token sequence of shape (batch, seq_len + generated)
        """
        self.eval()

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Crop to block size if needed
            idx_cond = input_ids
            if input_ids.size(1) > self.config.block_size:
                idx_cond = input_ids[:, -self.config.block_size :]

            # Get logits for the last position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Greedy decode when temperature is 0 (or negative).
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / float(temperature)

                # Optional top-k filtering (k <= 0 disables).
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Optional top-p (nucleus) filtering.
                # Match Rust behavior: enable only for 0 <= top_p < 1.
                if top_p is not None and 0.0 <= float(top_p) < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above threshold.
                    sorted_indices_to_remove = cumulative_probs > float(top_p)
                    # Keep the first token above threshold.
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = False

                    # Scatter back to original indices.
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample from distribution.
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            # Note: pos_encoding also has parameters
            n_params -= sum(p.numel() for p in self.pos_encoding.parameters())
        return n_params


def create_model(config: Optional[AsciiGPTConfig] = None) -> AsciiGPT:
    """
    Create an AsciiGPT model with the given configuration.

    Args:
        config: Model configuration. If None, uses default config.

    Returns:
        Initialized AsciiGPT model
    """
    if config is None:
        config = AsciiGPTConfig()
    return AsciiGPT(config)


# Preset configurations for different model sizes
def get_small_config() -> AsciiGPTConfig:
    """~10M parameter model for fast iteration."""
    return AsciiGPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=1024,
    )


def get_medium_config() -> AsciiGPTConfig:
    """~20M parameter model (default)."""
    return AsciiGPTConfig(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=2048,
    )


def get_large_config() -> AsciiGPTConfig:
    """~30M parameter model for best quality."""
    return AsciiGPTConfig(
        n_layer=8,
        n_head=8,
        n_embd=512,
        block_size=4096,
    )


if __name__ == "__main__":  # pragma: no cover
    # Quick test
    print("Testing AsciiGPT Model")
    print("=" * 50)

    # Test configuration
    config = get_small_config()
    print(f"\nSmall config estimated params: {config.estimate_params():,}")

    config = get_medium_config()
    print(f"Medium config estimated params: {config.estimate_params():,}")

    config = get_large_config()
    print(f"Large config estimated params: {config.estimate_params():,}")

    # Create model
    print("\nCreating small model...")
    config = get_small_config()
    model = create_model(config)

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, labels=input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    print("\nTesting generation...")
    prompt = torch.tensor([[1, 52, 53, 54, 4]])  # BOS + some chars + SEP
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=40)
    print(f"Prompt length: {prompt.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")
    print(f"Generated tokens: {generated[0].tolist()}")

    print("\n" + "=" * 50)
    print("All tests passed!")
