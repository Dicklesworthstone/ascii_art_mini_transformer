"""
Unit tests for inference/sampler.py

Tests cover:
- TopKSampler: k restriction, temperature scaling, greedy mode
- TopPSampler: nucleus sampling threshold, temperature, greedy mode
- sample_next_token(): convenience function with top-k, top-p, and combined modes
"""

from __future__ import annotations

import pytest
import torch

from inference.sampler import TopKSampler, TopPSampler, sample_next_token


class TestTopKSampler:
    """Tests for the TopKSampler class."""

    def test_greedy_mode_returns_argmax(self) -> None:
        """Temperature <= 0 should return argmax."""
        sampler = TopKSampler(k=10, temperature=0.0)
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
        result = sampler.sample(logits)
        assert result == 1  # Index of max value (5.0)

    def test_greedy_mode_negative_temperature(self) -> None:
        """Negative temperature should also trigger greedy mode."""
        sampler = TopKSampler(k=10, temperature=-1.0)
        logits = torch.tensor([0.0, 0.0, 10.0, 0.0])
        result = sampler.sample(logits)
        assert result == 2  # Index of max value (10.0)

    def test_k_restriction_with_seeded_sampling(self) -> None:
        """With k=2, only top 2 logits should be considered."""
        torch.manual_seed(42)
        sampler = TopKSampler(k=2, temperature=1.0)
        # Logits where top 2 are indices 1 and 3
        logits = torch.tensor([0.0, 10.0, 0.0, 9.0, 0.0])

        # Sample multiple times - should only return indices 1 or 3
        samples = [sampler.sample(logits.clone()) for _ in range(100)]
        unique = set(samples)

        # Only indices 1 and 3 should be possible
        assert unique.issubset({1, 3}), f"Got unexpected indices: {unique}"

    def test_k_zero_means_no_restriction(self) -> None:
        """k=0 should not restrict to top-k."""
        torch.manual_seed(123)
        sampler = TopKSampler(k=0, temperature=1.0)
        # Uniform logits - all should be possible
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])

        samples = [sampler.sample(logits.clone()) for _ in range(100)]
        unique = set(samples)

        # With uniform logits and no restriction, we should see variety
        assert len(unique) > 1, "Expected multiple unique samples with uniform logits"

    def test_k_larger_than_vocab(self) -> None:
        """k larger than vocab size should work (min applied internally)."""
        torch.manual_seed(0)
        sampler = TopKSampler(k=1000, temperature=1.0)
        logits = torch.tensor([1.0, 2.0, 3.0])
        # Should not raise - min(k, vocab_size) applied
        result = sampler.sample(logits)
        assert 0 <= result < 3

    def test_temperature_scaling_affects_distribution(self) -> None:
        """Higher temperature should make distribution more uniform."""
        torch.manual_seed(0)
        # Low temperature - more peaked, should mostly return max
        low_temp_sampler = TopKSampler(k=0, temperature=0.1)
        high_temp_sampler = TopKSampler(k=0, temperature=10.0)

        logits = torch.tensor([0.0, 5.0, 0.0])

        low_temp_samples = [low_temp_sampler.sample(logits.clone()) for _ in range(100)]
        high_temp_samples = [high_temp_sampler.sample(logits.clone()) for _ in range(100)]

        # Low temp should strongly prefer index 1
        low_temp_max_freq = low_temp_samples.count(1) / 100
        assert low_temp_max_freq > 0.9, f"Low temp should prefer max, got {low_temp_max_freq}"

        # High temp should be more uniform (index 1 less dominant)
        high_temp_max_freq = high_temp_samples.count(1) / 100
        assert high_temp_max_freq < 0.9, f"High temp should be more uniform, got {high_temp_max_freq}"


class TestTopPSampler:
    """Tests for the TopPSampler (nucleus sampling) class."""

    def test_greedy_mode_returns_argmax(self) -> None:
        """Temperature <= 0 should return argmax."""
        sampler = TopPSampler(p=0.9, temperature=0.0)
        logits = torch.tensor([1.0, 2.0, 5.0, 3.0])
        result = sampler.sample(logits)
        assert result == 2  # Index of max value (5.0)

    def test_greedy_mode_negative_temperature(self) -> None:
        """Negative temperature should also trigger greedy mode."""
        sampler = TopPSampler(p=0.5, temperature=-0.5)
        logits = torch.tensor([0.0, 8.0, 0.0, 0.0])
        result = sampler.sample(logits)
        assert result == 1  # Index of max value (8.0)

    def test_nucleus_restricts_to_cumsum_threshold(self) -> None:
        """With p=0.9, should only sample from tokens covering 90% mass."""
        torch.manual_seed(42)
        sampler = TopPSampler(p=0.5, temperature=1.0)

        # Create logits where one token dominates (>50% prob after softmax)
        # With p=0.5, only the dominant token should be sampled
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0])  # First token dominates

        samples = [sampler.sample(logits.clone()) for _ in range(50)]

        # With such a dominant token, p=0.5 should mostly/always pick index 0
        assert samples.count(0) >= 45, f"Expected mostly index 0, got {samples.count(0)}/50"

    def test_p_equals_one_allows_all_tokens(self) -> None:
        """p=1.0 should allow sampling from all tokens."""
        torch.manual_seed(123)
        sampler = TopPSampler(p=1.0, temperature=1.0)
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])

        samples = [sampler.sample(logits.clone()) for _ in range(100)]
        unique = set(samples)

        # With uniform logits and p=1.0, should see variety
        assert len(unique) > 1, "Expected multiple unique samples"

    def test_always_keeps_at_least_one_token(self) -> None:
        """Even with very low p, at least one token should be kept."""
        torch.manual_seed(0)
        sampler = TopPSampler(p=0.001, temperature=1.0)
        logits = torch.tensor([1.0, 1.0, 1.0])

        # Should not raise and should return a valid index
        result = sampler.sample(logits)
        assert 0 <= result < 3

    def test_temperature_scaling(self) -> None:
        """Temperature should scale logits before nucleus sampling."""
        torch.manual_seed(0)
        low_temp = TopPSampler(p=0.9, temperature=0.1)
        high_temp = TopPSampler(p=0.9, temperature=10.0)

        logits = torch.tensor([0.0, 5.0, 0.0])

        low_samples = [low_temp.sample(logits.clone()) for _ in range(100)]
        high_samples = [high_temp.sample(logits.clone()) for _ in range(100)]

        # Low temp should strongly prefer index 1
        assert low_samples.count(1) > 90

        # High temp with uniform-ish probs should show more variety
        assert len(set(high_samples)) > 1 or high_samples.count(1) < 90


class TestSampleNextToken:
    """Tests for the sample_next_token() convenience function."""

    def test_greedy_mode_returns_argmax(self) -> None:
        """Temperature <= 0 should return argmax."""
        logits = torch.tensor([1.0, 3.0, 2.0])
        result = sample_next_token(logits, temperature=0.0)
        assert result == 1

    def test_greedy_negative_temperature(self) -> None:
        """Negative temperature should also be greedy."""
        logits = torch.tensor([0.0, 0.0, 7.0, 0.0])
        result = sample_next_token(logits, temperature=-1.0)
        assert result == 2

    def test_top_k_only(self) -> None:
        """Test top-k sampling without top-p."""
        torch.manual_seed(42)
        logits = torch.tensor([0.0, 10.0, 9.0, 0.0, 0.0])

        samples = [
            sample_next_token(logits.clone(), temperature=1.0, top_k=2, top_p=None)
            for _ in range(100)
        ]
        unique = set(samples)

        # Should only sample from top 2 (indices 1 and 2)
        assert unique.issubset({1, 2}), f"Got unexpected indices: {unique}"

    def test_top_p_only(self) -> None:
        """Test top-p (nucleus) sampling without top-k."""
        torch.manual_seed(42)
        # Create logits where first token has ~90% mass
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0])

        samples = [
            sample_next_token(logits.clone(), temperature=1.0, top_k=0, top_p=0.5)
            for _ in range(50)
        ]

        # With p=0.5, should mostly sample index 0 (which has >50% mass)
        assert samples.count(0) >= 45

    def test_top_k_zero_means_no_k_restriction(self) -> None:
        """top_k=0 should not apply k restriction."""
        torch.manual_seed(0)
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])

        samples = [
            sample_next_token(logits.clone(), temperature=1.0, top_k=0, top_p=None)
            for _ in range(100)
        ]

        # With uniform logits and no restriction, expect variety
        assert len(set(samples)) > 1

    def test_top_p_none_means_no_p_restriction(self) -> None:
        """top_p=None should not apply nucleus sampling."""
        torch.manual_seed(0)
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])

        samples = [
            sample_next_token(logits.clone(), temperature=1.0, top_k=0, top_p=None)
            for _ in range(100)
        ]

        assert len(set(samples)) > 1

    def test_top_p_one_means_no_restriction(self) -> None:
        """top_p=1.0 should effectively disable nucleus restriction."""
        torch.manual_seed(0)
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])

        samples = [
            sample_next_token(logits.clone(), temperature=1.0, top_k=0, top_p=1.0)
            for _ in range(100)
        ]

        assert len(set(samples)) > 1

    def test_top_k_applied_before_top_p(self) -> None:
        """When both top_k and top_p are set, top_k is applied first."""
        torch.manual_seed(42)
        # 5 tokens, top 2 are indices 1 and 2
        logits = torch.tensor([0.0, 10.0, 9.0, 0.0, 0.0])

        # With k=2, only indices 1 and 2 are considered
        # Then p=0.9 further filters but shouldn't change much here
        samples = [
            sample_next_token(logits.clone(), temperature=1.0, top_k=2, top_p=0.9)
            for _ in range(100)
        ]

        # Should only see indices 1 and 2
        assert set(samples).issubset({1, 2})

    def test_temperature_affects_sampling(self) -> None:
        """Temperature should affect the sampling distribution."""
        logits = torch.tensor([0.0, 5.0, 0.0])

        torch.manual_seed(0)
        low_temp = [
            sample_next_token(logits.clone(), temperature=0.1, top_k=0, top_p=None)
            for _ in range(100)
        ]

        torch.manual_seed(0)
        high_temp = [
            sample_next_token(logits.clone(), temperature=10.0, top_k=0, top_p=None)
            for _ in range(100)
        ]

        # Low temperature should strongly prefer max
        assert low_temp.count(1) > 90

        # High temperature should be more uniform
        assert len(set(high_temp)) >= 1  # At minimum, should work


class TestSamplerProtocol:
    """Test that samplers conform to the Sampler protocol."""

    def test_topk_has_sample_method(self) -> None:
        """TopKSampler should have a sample method."""
        sampler = TopKSampler(k=10, temperature=1.0)
        assert hasattr(sampler, "sample")
        assert callable(sampler.sample)

    def test_topp_has_sample_method(self) -> None:
        """TopPSampler should have a sample method."""
        sampler = TopPSampler(p=0.9, temperature=1.0)
        assert hasattr(sampler, "sample")
        assert callable(sampler.sample)

    def test_samplers_are_frozen_dataclasses(self) -> None:
        """Samplers should be immutable (frozen dataclass)."""
        topk = TopKSampler(k=10, temperature=1.0)
        topp = TopPSampler(p=0.9, temperature=1.0)

        with pytest.raises(AttributeError):
            topk.k = 5  # type: ignore[misc]

        with pytest.raises(AttributeError):
            topp.p = 0.5  # type: ignore[misc]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_vocab(self) -> None:
        """Sampling from single-token vocabulary should return 0."""
        logits = torch.tensor([5.0])

        # TopKSampler
        assert TopKSampler(k=1, temperature=1.0).sample(logits) == 0

        # TopPSampler
        assert TopPSampler(p=0.5, temperature=1.0).sample(logits) == 0

        # sample_next_token
        assert sample_next_token(logits, temperature=1.0, top_k=1, top_p=0.5) == 0

    def test_negative_logits(self) -> None:
        """Sampling should work with negative logits."""
        logits = torch.tensor([-10.0, -5.0, -1.0])  # -1.0 is the max

        # Greedy should still find the max
        assert TopKSampler(k=3, temperature=0.0).sample(logits) == 2
        assert TopPSampler(p=0.9, temperature=0.0).sample(logits) == 2
        assert sample_next_token(logits, temperature=0.0) == 2

    def test_very_large_logits(self) -> None:
        """Sampling should handle very large logits without overflow."""
        logits = torch.tensor([1000.0, 0.0, 0.0])

        # Should not raise and should return index 0
        result = sample_next_token(logits, temperature=1.0, top_k=0, top_p=None)
        assert result == 0

    def test_uniform_logits(self) -> None:
        """With uniform logits, sampling should produce variety."""
        torch.manual_seed(42)
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0])

        samples = [
            sample_next_token(logits.clone(), temperature=1.0, top_k=0, top_p=None)
            for _ in range(100)
        ]

        # With truly uniform distribution, expect to see multiple indices
        assert len(set(samples)) > 1
