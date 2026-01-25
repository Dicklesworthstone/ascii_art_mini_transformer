"""
Unit tests for the ASCII art tokenizer.
"""

import json
import tempfile

import pytest

from python.model.tokenizer import (
    AsciiTokenizer,
    NUM_SPECIAL_TOKENS,
    PRINTABLE_ASCII,
    SPECIAL_TOKENS,
    STYLE_TOKENS,
    get_tokenizer,
)


class TestVocabulary:
    """Tests for vocabulary construction."""

    def test_vocab_size(self):
        """Vocab size should be special tokens + printable ASCII."""
        tokenizer = AsciiTokenizer()
        expected = NUM_SPECIAL_TOKENS + len(PRINTABLE_ASCII)
        assert tokenizer.vocab_size == expected
        assert tokenizer.vocab_size == 107

    def test_special_tokens_at_start(self):
        """Special tokens should have IDs 0 to NUM_SPECIAL_TOKENS-1."""
        tokenizer = AsciiTokenizer()
        for name, expected_id in SPECIAL_TOKENS.items():
            assert tokenizer.get_special_token_id(name) == expected_id
            assert expected_id < NUM_SPECIAL_TOKENS

    def test_printable_ascii_after_special(self):
        """Printable ASCII should start after special tokens."""
        tokenizer = AsciiTokenizer()
        # Space (first printable char) should be at offset
        space_id = tokenizer.encode(' ')[0]
        assert space_id == NUM_SPECIAL_TOKENS
        # Tilde (last printable char) should be at end
        tilde_id = tokenizer.encode('~')[0]
        assert tilde_id == NUM_SPECIAL_TOKENS + 94  # 95 chars, 0-indexed

    def test_all_printable_ascii_encoded(self):
        """Every printable ASCII char should have a unique ID."""
        tokenizer = AsciiTokenizer()
        ids_seen = set()
        for char in PRINTABLE_ASCII:
            tokens = tokenizer.encode(char)
            assert len(tokens) == 1
            token_id = tokens[0]
            assert token_id not in ids_seen
            ids_seen.add(token_id)


class TestBasicEncoding:
    """Tests for basic text encoding."""

    def test_encode_empty_string(self):
        """Empty string should encode to empty list."""
        tokenizer = AsciiTokenizer()
        assert tokenizer.encode('') == []

    def test_encode_single_char(self):
        """Single char should encode to single token."""
        tokenizer = AsciiTokenizer()
        assert len(tokenizer.encode('A')) == 1
        assert len(tokenizer.encode(' ')) == 1
        assert len(tokenizer.encode('~')) == 1

    def test_encode_with_special_tokens(self):
        """add_special_tokens should wrap with BOS/EOS."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode('Hi', add_special_tokens=True)
        assert tokens[0] == tokenizer.bos_token_id
        assert tokens[-1] == tokenizer.eos_token_id
        assert len(tokens) == 4  # BOS + 'H' + 'i' + EOS

    def test_encode_unknown_char(self):
        """Unknown chars should encode to UNK token."""
        tokenizer = AsciiTokenizer()
        # Unicode char not in ASCII
        tokens = tokenizer.encode('\u263a')  # Smiley face
        assert tokens[0] == tokenizer.unk_token_id

    def test_encode_preserves_spaces(self):
        """Spaces should be preserved exactly."""
        tokenizer = AsciiTokenizer()
        text = 'a b  c   d'
        decoded = tokenizer.decode(tokenizer.encode(text))
        assert decoded == text


class TestBasicDecoding:
    """Tests for basic decoding."""

    def test_decode_empty_list(self):
        """Empty list should decode to empty string."""
        tokenizer = AsciiTokenizer()
        assert tokenizer.decode([]) == ''

    def test_decode_skip_special_tokens(self):
        """By default, special tokens should be skipped."""
        tokenizer = AsciiTokenizer()
        tokens = [tokenizer.bos_token_id, 52, 53, tokenizer.eos_token_id]  # BOS + Hi + EOS
        decoded = tokenizer.decode(tokens)
        assert '<BOS>' not in decoded
        assert '<EOS>' not in decoded

    def test_decode_include_special_tokens(self):
        """Can include special tokens in output."""
        tokenizer = AsciiTokenizer()
        tokens = [tokenizer.bos_token_id, 52]  # BOS + 'H'
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        assert '<BOS>' in decoded

    def test_decode_newline_token(self):
        """NEWLINE token should decode to actual newline."""
        tokenizer = AsciiTokenizer()
        tokens = [52, tokenizer.newline_token_id, 53]  # H + newline + I
        decoded = tokenizer.decode(tokens)
        assert '\n' in decoded


class TestRoundTrip:
    """Tests for encode/decode round-trip fidelity."""

    def test_roundtrip_simple_text(self):
        """Simple text should round-trip perfectly."""
        tokenizer = AsciiTokenizer()
        for text in ['Hello', 'World!', 'Test 123', 'a b c']:
            assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_roundtrip_all_printable_ascii(self):
        """All printable ASCII chars should round-trip."""
        tokenizer = AsciiTokenizer()
        text = ''.join(PRINTABLE_ASCII)
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_roundtrip_special_chars(self):
        """Special characters in ASCII should round-trip."""
        tokenizer = AsciiTokenizer()
        text = '!@#$%^&*()_+-=[]{}|;:\'",.<>?/`~'
        assert tokenizer.decode(tokenizer.encode(text)) == text


class TestArtEncoding:
    """Tests for ASCII art-specific encoding."""

    def test_encode_art_converts_newlines(self):
        """Newlines in art should become NEWLINE tokens."""
        tokenizer = AsciiTokenizer()
        art = "line1\nline2\nline3"
        tokens = tokenizer._encode_art(art)
        newline_count = tokens.count(tokenizer.newline_token_id)
        assert newline_count == 2

    def test_decode_art_preserves_structure(self):
        """Decoded art should preserve newline structure."""
        tokenizer = AsciiTokenizer()
        art = " /\\_/\\\n( o.o )\n > ^ <"
        tokens = tokenizer._encode_art(art)
        decoded = tokenizer.decode(tokens)
        assert decoded == art


class TestTrainingExample:
    """Tests for training example encoding."""

    def test_training_example_structure(self):
        """Training example should have correct token structure."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode_training_example(
            description='a cat',
            art='meow',
            width=10,
            height=5,
            style='art',
        )
        # Should start with BOS
        assert tokens[0] == tokenizer.bos_token_id
        # Should end with EOS
        assert tokens[-1] == tokenizer.eos_token_id
        # Should contain SEP
        assert tokenizer.sep_token_id in tokens
        # Should contain WIDTH and HEIGHT markers
        assert SPECIAL_TOKENS['<WIDTH>'] in tokens
        assert SPECIAL_TOKENS['<HEIGHT>'] in tokens
        # Should contain style token
        assert SPECIAL_TOKENS['<STYLE_ART>'] in tokens

    def test_training_example_without_constraints(self):
        """Training example without constraints should work."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode_training_example(
            description='test',
            art='art',
        )
        # Should not have WIDTH/HEIGHT markers
        assert SPECIAL_TOKENS['<WIDTH>'] not in tokens
        assert SPECIAL_TOKENS['<HEIGHT>'] not in tokens

    def test_training_example_all_styles(self):
        """All style types should work."""
        tokenizer = AsciiTokenizer()
        for style, token_name in STYLE_TOKENS.items():
            tokens = tokenizer.encode_training_example(
                description='test',
                art='art',
                style=style,
            )
            assert SPECIAL_TOKENS[token_name] in tokens


class TestInferencePrompt:
    """Tests for inference prompt encoding."""

    def test_inference_prompt_ends_with_sep(self):
        """Inference prompt should end with SEP token."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode_inference_prompt(
            description='a cat',
            width=40,
        )
        assert tokens[-1] == tokenizer.sep_token_id

    def test_inference_prompt_starts_with_bos(self):
        """Inference prompt should start with BOS."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode_inference_prompt(description='test')
        assert tokens[0] == tokenizer.bos_token_id

    def test_inference_prompt_no_eos(self):
        """Inference prompt should NOT have EOS (model generates after)."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode_inference_prompt(description='test')
        assert tokenizer.eos_token_id not in tokens


class TestDecodeArt:
    """Tests for decode_art method."""

    def test_decode_art_finds_separator(self):
        """decode_art should find and skip past SEP."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode_training_example(
            description='cat',
            art='meow',
        )
        art = tokenizer.decode_art(tokens)
        assert art == 'meow'
        assert 'cat' not in art

    def test_decode_art_stops_at_eos(self):
        """decode_art should stop at EOS token."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode_training_example(
            description='desc',
            art='art',
        )
        # Add extra tokens after EOS (shouldn't appear)
        tokens.extend([52, 53, 54])
        art = tokenizer.decode_art(tokens)
        assert art == 'art'

    def test_decode_art_no_separator(self):
        """decode_art without SEP should decode everything."""
        tokenizer = AsciiTokenizer()
        tokens = tokenizer.encode('hello')
        art = tokenizer.decode_art(tokens)
        assert art == 'hello'


class TestSpecialTokenHelpers:
    """Tests for special token helper methods."""

    def test_is_special_token(self):
        """is_special_token should identify special tokens correctly."""
        tokenizer = AsciiTokenizer()
        # Special tokens
        assert tokenizer.is_special_token(tokenizer.pad_token_id)
        assert tokenizer.is_special_token(tokenizer.bos_token_id)
        assert tokenizer.is_special_token(tokenizer.eos_token_id)
        # Non-special tokens (printable chars)
        assert not tokenizer.is_special_token(NUM_SPECIAL_TOKENS)
        assert not tokenizer.is_special_token(NUM_SPECIAL_TOKENS + 50)

    def test_get_special_token_id_invalid(self):
        """get_special_token_id should raise for unknown tokens."""
        tokenizer = AsciiTokenizer()
        with pytest.raises(ValueError):
            tokenizer.get_special_token_id('<INVALID>')


class TestSaveLoad:
    """Tests for tokenizer save/load."""

    def test_save_creates_json(self):
        """save should create valid JSON file."""
        tokenizer = AsciiTokenizer()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        tokenizer.save(path)

        with open(path) as f:
            config = json.load(f)

        assert config['vocab_size'] == tokenizer.vocab_size
        assert config['num_special_tokens'] == NUM_SPECIAL_TOKENS
        assert 'special_tokens' in config
        assert 'style_tokens' in config

        # Intentionally do not delete temp files from tests; project policy forbids deletion.

    def test_load_returns_valid_tokenizer(self):
        """load should return a working tokenizer."""
        tokenizer = AsciiTokenizer()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        tokenizer.save(path)
        loaded = AsciiTokenizer.load(path)

        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.encode('test') == tokenizer.encode('test')

        # Intentionally do not delete temp files from tests; project policy forbids deletion.


class TestEdgeCases:
    """Tests for edge cases."""

    def test_max_length_text(self):
        """Long text should encode without issues."""
        tokenizer = AsciiTokenizer()
        text = 'A' * 10000
        tokens = tokenizer.encode(text)
        assert len(tokens) == 10000
        assert tokenizer.decode(tokens) == text

    def test_repeated_special_chars(self):
        """Repeated special chars should work."""
        tokenizer = AsciiTokenizer()
        text = '!!!???...//'
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_numbers_only(self):
        """Numbers should encode correctly."""
        tokenizer = AsciiTokenizer()
        text = '0123456789'
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_empty_lines_in_art(self):
        """Empty lines in art should be preserved."""
        tokenizer = AsciiTokenizer()
        art = "line1\n\nline3"  # Empty middle line
        tokens = tokenizer._encode_art(art)
        decoded = tokenizer.decode(tokens)
        assert decoded == art
        assert '\n\n' in decoded


class TestGetTokenizer:
    """Tests for get_tokenizer convenience function."""

    def test_get_tokenizer_returns_instance(self):
        """get_tokenizer should return valid tokenizer."""
        tokenizer = get_tokenizer()
        assert isinstance(tokenizer, AsciiTokenizer)
        assert tokenizer.vocab_size == 107


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
