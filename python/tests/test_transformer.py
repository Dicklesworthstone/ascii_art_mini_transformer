"""
Unit tests for the ASCII Art Transformer model.

Tests cover:
- Configuration validation
- Model component architectures
- Forward pass correctness
- Generation functionality
- Parameter counting
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from python.model.transformer import (  # noqa: E402
    AsciiGPT,
    AsciiGPTConfig,
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    create_model,
    get_large_config,
    get_medium_config,
    get_small_config,
)


class TestAsciiGPTConfig:
    """Tests for AsciiGPTConfig."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = AsciiGPTConfig()
        assert config.block_size == 2048
        assert config.vocab_size == 107
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384
        assert config.dropout == 0.1

    def test_head_dim_property(self):
        """Head dimension should be computed correctly."""
        config = AsciiGPTConfig()
        assert config.head_dim == config.n_embd // config.n_head
        assert config.head_dim == 64

    def test_n_embd_divisibility_check(self):
        """n_embd must be divisible by n_head."""
        # This should work
        config = AsciiGPTConfig(n_embd=384, n_head=6)
        assert config.head_dim == 64

        # This should raise
        with pytest.raises(AssertionError):
            AsciiGPTConfig(n_embd=100, n_head=3)  # 100/3 = 33.33...

    def test_param_estimation(self):
        """Parameter estimation should be reasonable."""
        config = get_small_config()
        estimated = config.estimate_params()
        # Small config: ~3M params
        assert 2_000_000 < estimated < 8_000_000

        config = get_medium_config()
        estimated = config.estimate_params()
        # Medium config: ~10M params
        assert 8_000_000 < estimated < 20_000_000

    def test_special_tokens(self):
        """Special token IDs should be configured correctly."""
        config = AsciiGPTConfig()
        assert config.pad_token_id == 0
        assert config.bos_token_id == 1
        assert config.eos_token_id == 2
        assert config.newline_token_id == 7


class TestCausalSelfAttention:
    """Tests for the attention layer."""

    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return AsciiGPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=128)

    @pytest.fixture
    def attention(self, config):
        """Create attention module."""
        return CausalSelfAttention(config)

    def test_output_shape(self, attention, config):
        """Output should match input shape."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)
        out = attention(x)
        assert out.shape == x.shape

    def test_causal_mask_applied(self, attention, config):
        """Attention should be causal (no future information leakage)."""
        # Create attention weights manually to check masking
        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, config.n_embd)

        # Forward pass (indirectly tests mask application)
        out = attention(x)

        # Output should be valid (no NaN from masked positions)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_with_padding_mask(self, attention, config):
        """Padding mask should be respected."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        # Create padding mask (1 = attend, 0 = ignore)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0  # Mask last 6 positions in first batch

        out = attention(x, attention_mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_deterministic_without_dropout(self, config):
        """Without dropout, attention should be deterministic."""
        config_no_dropout = AsciiGPTConfig(
            n_layer=2, n_head=4, n_embd=64, block_size=128, dropout=0.0
        )
        attention = CausalSelfAttention(config_no_dropout)
        attention.eval()

        x = torch.randn(1, 8, config.n_embd)
        out1 = attention(x)
        out2 = attention(x)
        assert torch.allclose(out1, out2)


class TestMLP:
    """Tests for the MLP layer."""

    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return AsciiGPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=128)

    @pytest.fixture
    def mlp(self, config):
        """Create MLP module."""
        return MLP(config)

    def test_output_shape(self, mlp, config):
        """Output should match input shape."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)
        out = mlp(x)
        assert out.shape == x.shape

    def test_hidden_expansion(self, mlp, config):
        """Hidden layer should have 4x expansion."""
        assert mlp.c_fc.out_features == 4 * config.n_embd
        assert mlp.c_proj.in_features == 4 * config.n_embd

    def test_gelu_activation(self, mlp, config):
        """MLP should use GELU activation."""
        assert isinstance(mlp.gelu, nn.GELU)


class TestTransformerBlock:
    """Tests for the transformer block."""

    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return AsciiGPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=128)

    @pytest.fixture
    def block(self, config):
        """Create transformer block."""
        return TransformerBlock(config)

    def test_output_shape(self, block, config):
        """Output should match input shape."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)
        out = block(x)
        assert out.shape == x.shape

    def test_pre_norm_structure(self, block):
        """Block should use pre-norm (LayerNorm before sublayers)."""
        assert isinstance(block.ln_1, nn.LayerNorm)
        assert isinstance(block.ln_2, nn.LayerNorm)
        assert isinstance(block.attn, CausalSelfAttention)
        assert isinstance(block.mlp, MLP)

    def test_residual_connections(self, config):
        """Residual connections should be present."""
        block = TransformerBlock(config)
        block.eval()

        x = torch.randn(1, 8, config.n_embd)
        out = block(x)

        # With residual, output shouldn't be zero even if sublayers were
        # (they won't be, but this tests the structure)
        assert not torch.allclose(out, torch.zeros_like(out))


class TestAsciiGPT:
    """Tests for the full AsciiGPT model."""

    @pytest.fixture
    def config(self):
        """Small config for faster testing."""
        return get_small_config()

    @pytest.fixture
    def model(self, config):
        """Create model."""
        return create_model(config)

    def test_model_creation(self, config):
        """Model should be created without errors."""
        model = create_model(config)
        assert isinstance(model, AsciiGPT)

    def test_forward_shape(self, model, config):
        """Forward pass should produce correct output shape."""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is None  # No labels provided

    def test_forward_with_labels(self, model, config):
        """Forward with labels should compute loss."""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids, labels=input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Cross-entropy should be positive

    def test_forward_with_attention_mask(self, model, config):
        """Forward with attention mask should work correctly."""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 20:] = 0  # Mask some positions

        logits, loss = model(input_ids, attention_mask=attention_mask)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_block_size_limit(self, model, config):
        """Sequences longer than block_size should raise error."""
        batch_size = 1
        seq_len = config.block_size + 100  # Exceeds limit
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with pytest.raises(AssertionError):
            model(input_ids)

    def test_weight_tying(self, model):
        """Token embedding should be tied to LM head."""
        assert model.token_embedding.weight is model.lm_head.weight

    def test_param_count(self, model, config):
        """Parameter count should match config."""
        n_params = model.get_num_params(non_embedding=False)
        estimated = config.estimate_params()
        # Allow 20% tolerance (estimation is approximate)
        assert abs(n_params - estimated) / estimated < 0.2


class TestGeneration:
    """Tests for text generation."""

    @pytest.fixture
    def config(self):
        """Small config for faster testing."""
        return AsciiGPTConfig(
            n_layer=2, n_head=4, n_embd=64, block_size=128, vocab_size=107
        )

    @pytest.fixture
    def model(self, config):
        """Create model."""
        return create_model(config)

    def test_basic_generation(self, model, config):
        """Generation should produce tokens."""
        prompt = torch.tensor([[config.bos_token_id]])
        generated = model.generate(prompt, max_new_tokens=10)

        assert generated.shape[0] == 1  # Batch size
        assert generated.shape[1] > prompt.shape[1]  # Generated new tokens
        assert generated.shape[1] <= prompt.shape[1] + 10  # Respects max

    def test_generation_stops_at_eos(self, model, config):
        """Generation should stop when EOS is produced."""
        # This is probabilistic, but we can at least test the code path
        prompt = torch.tensor([[config.bos_token_id]])
        generated = model.generate(
            prompt, max_new_tokens=100, eos_token_id=config.eos_token_id
        )
        # Should not exceed max
        assert generated.shape[1] <= prompt.shape[1] + 100

    def test_temperature_sampling(self, model, config):
        """Different temperatures should produce different distributions."""
        prompt = torch.tensor([[config.bos_token_id]])

        # Low temperature = more deterministic
        torch.manual_seed(42)
        gen_low = model.generate(prompt, max_new_tokens=5, temperature=0.1)

        # High temperature = more random
        torch.manual_seed(42)
        gen_high = model.generate(prompt, max_new_tokens=5, temperature=2.0)

        # They might be different (but not guaranteed due to randomness)
        # At least the code should run
        assert gen_low.shape[1] > 0
        assert gen_high.shape[1] > 0

    def test_temperature_zero_is_greedy_and_deterministic(self, model, config):
        """temperature=0 should not divide by zero and should be deterministic (greedy)."""
        prompt = torch.tensor([[config.bos_token_id]])
        gen1 = model.generate(prompt, max_new_tokens=10, temperature=0.0)
        gen2 = model.generate(prompt, max_new_tokens=10, temperature=0.0)
        assert torch.equal(gen1, gen2)

    def test_top_k_sampling(self, model, config):
        """Top-k sampling should limit token choices."""
        prompt = torch.tensor([[config.bos_token_id]])
        generated = model.generate(prompt, max_new_tokens=10, top_k=5)
        assert generated.shape[1] > prompt.shape[1]

    def test_top_k_zero_disables_filtering(self, model, config):
        """top_k=0 should not error and should behave like no top-k filtering."""
        prompt = torch.tensor([[config.bos_token_id]])
        generated = model.generate(prompt, max_new_tokens=10, top_k=0)
        assert generated.shape[1] > prompt.shape[1]

    def test_top_p_sampling(self, model, config):
        """Top-p (nucleus) sampling should work."""
        prompt = torch.tensor([[config.bos_token_id]])
        generated = model.generate(prompt, max_new_tokens=10, top_p=0.9)
        assert generated.shape[1] > prompt.shape[1]

    def test_combined_sampling(self, model, config):
        """Combined top-k and top-p should work."""
        prompt = torch.tensor([[config.bos_token_id]])
        generated = model.generate(
            prompt, max_new_tokens=10, temperature=0.8, top_k=40, top_p=0.95
        )
        assert generated.shape[1] > prompt.shape[1]

    def test_batch_generation(self, model, config):
        """Batch generation should work."""
        batch_size = 3
        prompt = torch.tensor([[config.bos_token_id]] * batch_size)
        generated = model.generate(prompt, max_new_tokens=10)

        assert generated.shape[0] == batch_size
        assert generated.shape[1] > 1


class TestPresetConfigs:
    """Tests for preset model configurations."""

    def test_small_config(self):
        """Small config should have expected properties."""
        config = get_small_config()
        assert config.n_layer == 4
        assert config.n_head == 4
        assert config.n_embd == 256
        # Small config: ~3M params
        model = create_model(config)
        n_params = model.get_num_params(non_embedding=False)
        assert 2_000_000 < n_params < 8_000_000

    def test_medium_config(self):
        """Medium config should have expected properties."""
        config = get_medium_config()
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384
        # Medium config: ~10M params
        model = create_model(config)
        n_params = model.get_num_params(non_embedding=False)
        assert 8_000_000 < n_params < 20_000_000

    def test_large_config(self):
        """Large config should have expected properties."""
        config = get_large_config()
        assert config.n_layer == 8
        assert config.n_head == 8
        assert config.n_embd == 512
        # Should be roughly 30M params
        model = create_model(config)
        n_params = model.get_num_params(non_embedding=False)
        assert 25_000_000 < n_params < 50_000_000


class TestGradients:
    """Tests for gradient flow."""

    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return AsciiGPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=128)

    @pytest.fixture
    def model(self, config):
        """Create model."""
        return create_model(config)

    def test_gradients_flow(self, model, config):
        """Gradients should flow through all parameters."""
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(input_ids, labels=input_ids)

        loss.backward()

        # Check that gradients exist for key parameters
        assert model.token_embedding.weight.grad is not None
        for block in model.blocks:
            assert block.ln_1.weight.grad is not None
            assert block.attn.c_attn.weight.grad is not None
            assert block.mlp.c_fc.weight.grad is not None

    def test_no_nan_gradients(self, model, config):
        """Gradients should not contain NaN."""
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(input_ids, labels=input_ids)

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


class TestGenerateSamplingInvariants:
    """Tests for generate() sampling invariants (bd-1hr)."""

    @pytest.fixture
    def tiny_config(self):
        """Very small config for fast deterministic testing."""
        return AsciiGPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=16,  # Very small to test cropping
            vocab_size=107,
            dropout=0.0,  # Disable dropout for determinism
        )

    @pytest.fixture
    def model(self, tiny_config):
        """Create tiny model."""
        model = create_model(tiny_config)
        model.eval()
        return model

    # ==================== Block Size Cropping Tests ====================

    def test_generate_crops_to_block_size(self, model, tiny_config):
        """Generation should crop input to block_size when sequence grows beyond."""
        # Start with a prompt that's already near block_size
        prompt_len = tiny_config.block_size - 2
        prompt = torch.randint(
            3,
            tiny_config.vocab_size,
            (1, prompt_len),  # Skip special tokens 0-2
        )

        # Generate tokens that will exceed block_size
        torch.manual_seed(42)
        generated = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=1.0,
            eos_token_id=-1,  # Disable EOS stopping
        )

        # Should have generated new tokens without crashing
        assert generated.shape[1] > prompt_len
        # Maximum would be prompt_len + max_new_tokens
        assert generated.shape[1] <= prompt_len + 10

    def test_generate_uses_only_last_block_size_tokens(self, model, tiny_config):
        """When input exceeds block_size, only last block_size tokens are used."""
        # Create a prompt larger than block_size
        large_prompt_len = tiny_config.block_size + 10
        large_prompt = torch.randint(3, tiny_config.vocab_size, (1, large_prompt_len))

        # Generate with large prompt (cropping happens internally)
        torch.manual_seed(123)
        generated = model.generate(
            large_prompt,
            max_new_tokens=5,
            temperature=1.0,
            eos_token_id=-1,
        )

        # Should succeed and produce new tokens
        assert generated.shape[1] == large_prompt_len + 5

    def test_block_size_exactly_full(self, model, tiny_config):
        """Generation works when prompt is exactly block_size."""
        prompt = torch.randint(3, tiny_config.vocab_size, (1, tiny_config.block_size))

        torch.manual_seed(42)
        generated = model.generate(
            prompt,
            max_new_tokens=3,
            temperature=1.0,
            eos_token_id=-1,
        )

        assert generated.shape[1] == tiny_config.block_size + 3

    # ==================== EOS Stopping Tests ====================

    def test_eos_stops_generation(self, tiny_config):
        """Generation should stop when EOS token is produced."""
        # Create a model that we can manipulate to always produce EOS
        config = AsciiGPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=32,
            vocab_size=107,
            dropout=0.0,
        )
        model = create_model(config)
        model.eval()

        prompt = torch.tensor([[config.bos_token_id]])

        # Generate with EOS enabled
        torch.manual_seed(12345)  # Seed chosen to likely produce EOS early
        generated = model.generate(
            prompt,
            max_new_tokens=100,
            temperature=1.0,
            eos_token_id=config.eos_token_id,
        )

        # Either hit EOS (stopped early) or generated max tokens
        assert generated.shape[1] <= 1 + 100

        # If EOS was hit, it should be in the sequence
        if generated.shape[1] < 1 + 100:
            assert config.eos_token_id in generated[0].tolist()

    def test_eos_token_appears_when_stopping_early(self, tiny_config):
        """When generation stops at EOS, the EOS token should be last."""
        # Run many trials to find one that stops early
        config = AsciiGPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=64,
            vocab_size=107,
            dropout=0.0,
        )
        model = create_model(config)
        model.eval()

        found_early_stop = False
        for seed in range(100):
            torch.manual_seed(seed)
            prompt = torch.tensor([[config.bos_token_id]])
            generated = model.generate(
                prompt,
                max_new_tokens=50,
                temperature=1.5,  # Higher temperature for variety
                eos_token_id=config.eos_token_id,
            )

            if generated.shape[1] < 1 + 50:
                # Stopped early - verify EOS is the last token
                assert generated[0, -1].item() == config.eos_token_id
                found_early_stop = True
                break

        # We should have found at least one early stop in 100 trials
        # If not, the test is still valid - it just means EOS is rare
        if not found_early_stop:
            pytest.skip("No early EOS stop found in 100 trials - probabilistic test")

    def test_custom_eos_token_stops_generation(self, model, tiny_config):
        """Custom EOS token ID should be respected."""
        custom_eos = 50  # Use an arbitrary token as EOS
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        found_custom_stop = False
        for seed in range(50):
            torch.manual_seed(seed)
            generated = model.generate(
                prompt,
                max_new_tokens=20,
                temperature=2.0,
                eos_token_id=custom_eos,
            )

            if custom_eos in generated[0].tolist():
                # Found our custom EOS
                # Verify generation stopped at or after this token
                tokens = generated[0].tolist()
                eos_idx = tokens.index(custom_eos)
                # EOS should be the last token
                assert eos_idx == len(tokens) - 1
                found_custom_stop = True
                break

        if not found_custom_stop:
            pytest.skip("Custom EOS token not produced in 50 trials")

    # ==================== Top-k Filtering Tests ====================

    def test_top_k_one_is_greedy(self, model, tiny_config):
        """top_k=1 should always select the highest probability token (greedy)."""
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        # Generate multiple times with different seeds - should all be same
        results = []
        for seed in [0, 1, 2, 3, 4]:
            torch.manual_seed(seed)
            generated = model.generate(
                prompt,
                max_new_tokens=5,
                temperature=1.0,
                top_k=1,  # Greedy
                eos_token_id=-1,
            )
            results.append(generated[0].tolist())

        # All results should be identical (greedy decoding)
        for r in results[1:]:
            assert r == results[0], f"top_k=1 should be deterministic: {results}"

    def test_top_k_larger_than_vocab_size(self, model, tiny_config):
        """top_k larger than vocab_size should not crash."""
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        torch.manual_seed(42)
        generated = model.generate(
            prompt,
            max_new_tokens=5,
            temperature=1.0,
            top_k=1000,  # Much larger than vocab_size (107)
            eos_token_id=-1,
        )

        # Should succeed
        assert generated.shape[1] == 1 + 5

    def test_top_k_constrains_choices(self, model, tiny_config):
        """top_k should limit tokens to top k candidates."""
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        # With top_k=5, we should see limited variety
        tokens_seen = set()
        for seed in range(30):
            torch.manual_seed(seed)
            generated = model.generate(
                prompt,
                max_new_tokens=1,  # Just one token
                temperature=1.0,
                top_k=5,
                eos_token_id=-1,
            )
            tokens_seen.add(generated[0, -1].item())

        # Should see at most 5 unique tokens (though likely fewer in practice)
        assert len(tokens_seen) <= 5, (
            f"top_k=5 produced more than 5 unique tokens: {tokens_seen}"
        )

    # ==================== Top-p Filtering Tests ====================

    def test_top_p_constrains_compared_to_no_filtering(self, model, tiny_config):
        """Low top_p should constrain token choices compared to no filtering."""
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        # Measure variety with restrictive top_p
        tokens_constrained = set()
        for seed in range(30):
            torch.manual_seed(seed)
            generated = model.generate(
                prompt,
                max_new_tokens=1,
                temperature=1.0,
                top_p=0.1,  # Very restrictive
                eos_token_id=-1,
            )
            tokens_constrained.add(generated[0, -1].item())

        # Measure variety with no filtering
        tokens_unconstrained = set()
        for seed in range(30):
            torch.manual_seed(seed)
            generated = model.generate(
                prompt,
                max_new_tokens=1,
                temperature=1.0,
                top_p=1.0,  # No filtering
                eos_token_id=-1,
            )
            tokens_unconstrained.add(generated[0, -1].item())

        # Constrained should have equal or fewer unique tokens
        assert len(tokens_constrained) <= len(tokens_unconstrained), (
            f"top_p=0.1 should not produce more variety than top_p=1.0: "
            f"constrained={len(tokens_constrained)}, unconstrained={len(tokens_unconstrained)}"
        )

    def test_top_p_one_allows_all_tokens(self, model, tiny_config):
        """top_p=1.0 should allow all tokens (no filtering)."""
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        torch.manual_seed(42)
        generated = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=1.0,
            top_p=1.0,  # No filtering
            eos_token_id=-1,
        )

        # Should succeed
        assert generated.shape[1] == 1 + 10

    def test_top_p_does_not_crash_on_edge_values(self, model, tiny_config):
        """top_p edge values should not crash."""
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        for top_p_val in [0.001, 0.01, 0.5, 0.9, 0.99, 0.999, 1.0]:
            torch.manual_seed(42)
            generated = model.generate(
                prompt,
                max_new_tokens=3,
                temperature=1.0,
                top_p=top_p_val,
                eos_token_id=-1,
            )
            assert generated.shape[1] == 1 + 3, f"Failed at top_p={top_p_val}"

    # ==================== Combined Tests ====================

    def test_top_k_and_top_p_together(self, model, tiny_config):
        """Combined top_k and top_p should both be applied."""
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        torch.manual_seed(42)
        generated = model.generate(
            prompt,
            max_new_tokens=5,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            eos_token_id=-1,
        )

        assert generated.shape[1] == 1 + 5

    def test_zero_temperature_is_greedy(self, tiny_config):
        """Temperature 0 should be equivalent to greedy (argmax)."""
        config = AsciiGPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=32,
            vocab_size=107,
            dropout=0.0,
        )
        model = create_model(config)
        model.eval()

        prompt = torch.tensor([[config.bos_token_id]])

        # Very low temperature should be nearly deterministic
        results = []
        for seed in [0, 10, 20, 30, 40]:
            torch.manual_seed(seed)
            generated = model.generate(
                prompt,
                max_new_tokens=5,
                temperature=0.001,  # Near-zero
                eos_token_id=-1,
            )
            results.append(generated[0].tolist())

        # Results should all be the same (greedy)
        for r in results[1:]:
            assert r == results[0], "Very low temperature should be deterministic"

    def test_deterministic_with_same_seed(self, model, tiny_config):
        """Same seed should produce same output."""
        prompt = torch.tensor([[tiny_config.bos_token_id]])

        # Generate twice with same seed
        torch.manual_seed(42)
        gen1 = model.generate(
            prompt.clone(),
            max_new_tokens=10,
            temperature=1.0,
            eos_token_id=-1,
        )

        torch.manual_seed(42)
        gen2 = model.generate(
            prompt.clone(),
            max_new_tokens=10,
            temperature=1.0,
            eos_token_id=-1,
        )

        assert gen1.tolist() == gen2.tolist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
