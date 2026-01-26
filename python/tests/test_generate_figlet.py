"""
Unit tests for FIGlet dataset generator helpers (bd-1nj).

Tests cover:
- _word_list() word set selection
- is_valid_output() width/height thresholds
- discover_fonts() directory traversal and sorting
- Optional integration test with actual figlet binary
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from python.data.generate_figlet import (
    COMMON_WORDS,
    EXTRA_WORDS,
    _word_list,
    discover_fonts,
    is_valid_output,
)


class TestWordList(unittest.TestCase):
    """Tests for _word_list() word set selection."""

    def test_base_returns_common_words(self) -> None:
        """'base' should return COMMON_WORDS list."""
        result = _word_list("base")
        self.assertEqual(result, COMMON_WORDS)
        self.assertIn("HELLO", result)
        self.assertIn("WORLD", result)

    def test_extra_returns_extra_words(self) -> None:
        """'extra' should return EXTRA_WORDS list."""
        result = _word_list("extra")
        self.assertEqual(result, EXTRA_WORDS)
        self.assertIn("CAT", result)
        self.assertIn("DOG", result)

    def test_all_returns_combined_words(self) -> None:
        """'all' should return combined COMMON_WORDS + EXTRA_WORDS."""
        result = _word_list("all")
        self.assertEqual(len(result), len(COMMON_WORDS) + len(EXTRA_WORDS))
        # Contains items from both lists
        self.assertIn("HELLO", result)  # from COMMON_WORDS
        self.assertIn("CAT", result)  # from EXTRA_WORDS

    def test_all_preserves_order(self) -> None:
        """'all' should have COMMON_WORDS first, then EXTRA_WORDS."""
        result = _word_list("all")
        # First part should match COMMON_WORDS
        self.assertEqual(result[: len(COMMON_WORDS)], COMMON_WORDS)
        # Second part should match EXTRA_WORDS
        self.assertEqual(result[len(COMMON_WORDS) :], EXTRA_WORDS)

    def test_invalid_kind_raises_value_error(self) -> None:
        """Invalid kind should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _word_list("invalid")
        self.assertIn("Unsupported word set", str(ctx.exception))
        self.assertIn("invalid", str(ctx.exception))

    def test_word_lists_are_non_empty(self) -> None:
        """All word lists should be non-empty."""
        self.assertGreater(len(_word_list("base")), 0)
        self.assertGreater(len(_word_list("extra")), 0)
        self.assertGreater(len(_word_list("all")), 0)


class TestIsValidOutput(unittest.TestCase):
    """Tests for is_valid_output() width/height thresholds."""

    def test_empty_string_is_invalid(self) -> None:
        """Empty string should be invalid."""
        self.assertFalse(is_valid_output(""))

    def test_whitespace_only_is_invalid(self) -> None:
        """Whitespace-only string should be invalid."""
        self.assertFalse(is_valid_output("   \n\t\n   "))

    def test_none_is_invalid(self) -> None:
        """None should be invalid."""
        # The function has `if not art` which handles None
        self.assertFalse(is_valid_output(None))  # type: ignore[arg-type]

    def test_simple_valid_output(self) -> None:
        """Simple ASCII art should be valid."""
        art = "HELLO\nWORLD"
        self.assertTrue(is_valid_output(art))

    def test_single_line_is_valid(self) -> None:
        """Single non-empty line should be valid."""
        art = "X"
        self.assertTrue(is_valid_output(art))

    def test_multi_line_figlet_style_is_valid(self) -> None:
        """Multi-line FIGlet-style output should be valid."""
        art = """\
  _   _ _____ _     _     ___
 | | | | ____| |   | |   / _ \\
 | |_| |  _| | |   | |  | | | |
 |  _  | |___| |___| |__| |_| |
 |_| |_|_____|_____|_____\\___/
"""
        self.assertTrue(is_valid_output(art))

    def test_too_wide_is_invalid(self) -> None:
        """Output wider than 200 chars should be invalid."""
        # Create a line that's exactly 201 chars
        art = "X" * 201
        self.assertFalse(is_valid_output(art))

    def test_exactly_200_width_is_valid(self) -> None:
        """Output exactly 200 chars wide should be valid."""
        art = "X" * 200
        self.assertTrue(is_valid_output(art))

    def test_too_tall_is_invalid(self) -> None:
        """Output taller than 50 lines should be invalid."""
        # Create 51 lines
        art = "\n".join(["X"] * 51)
        self.assertFalse(is_valid_output(art))

    def test_exactly_50_lines_is_valid(self) -> None:
        """Output exactly 50 lines should be valid."""
        art = "\n".join(["X"] * 50)
        self.assertTrue(is_valid_output(art))

    def test_trailing_newlines_counted_in_height(self) -> None:
        """Trailing newlines should count towards line count."""
        # 49 content lines + many empty lines at end = > 50 total lines
        art = "\n".join(["X"] * 49) + "\n" * 10
        # This creates 49 + 10 = 59 lines total
        self.assertFalse(is_valid_output(art))

    def test_only_empty_lines_is_invalid(self) -> None:
        """All-empty lines (but with newlines) should be invalid."""
        art = "\n\n\n\n"
        self.assertFalse(is_valid_output(art))


class TestDiscoverFonts(unittest.TestCase):
    """Tests for discover_fonts() directory traversal and sorting."""

    def test_empty_directory_returns_empty_list(self) -> None:
        """Empty directory should return empty list."""
        with TemporaryDirectory() as tmpdir:
            result = discover_fonts(Path(tmpdir))
            self.assertEqual(result, [])

    def test_finds_flf_files(self) -> None:
        """Should find .flf files in directory."""
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create some .flf files
            (tmppath / "font1.flf").touch()
            (tmppath / "font2.flf").touch()

            result = discover_fonts(tmppath)
            self.assertEqual(len(result), 2)

    def test_ignores_non_flf_files(self) -> None:
        """Should ignore non-.flf files."""
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "font.flf").touch()
            (tmppath / "readme.txt").touch()
            (tmppath / "font.ttf").touch()

            result = discover_fonts(tmppath)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "font.flf")

    def test_finds_fonts_in_subdirectories(self) -> None:
        """Should recursively find .flf files in subdirectories."""
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create subdirectory structure
            subdir1 = tmppath / "contrib"
            subdir2 = tmppath / "contrib" / "nested"
            subdir1.mkdir()
            subdir2.mkdir()

            (tmppath / "root.flf").touch()
            (subdir1 / "contrib1.flf").touch()
            (subdir2 / "nested.flf").touch()

            result = discover_fonts(tmppath)
            self.assertEqual(len(result), 3)

    def test_returns_sorted_list(self) -> None:
        """Should return fonts sorted by path."""
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create fonts in non-alphabetical order
            (tmppath / "zebra.flf").touch()
            (tmppath / "alpha.flf").touch()
            (tmppath / "beta.flf").touch()

            result = discover_fonts(tmppath)
            names = [p.name for p in result]
            self.assertEqual(names, ["alpha.flf", "beta.flf", "zebra.flf"])

    def test_returns_path_objects(self) -> None:
        """Should return Path objects, not strings."""
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "font.flf").touch()

            result = discover_fonts(tmppath)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], Path)

    def test_handles_nonexistent_directory(self) -> None:
        """Non-existent directory should return empty list (rglob behavior)."""
        with TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"
            # pathlib rglob on non-existent dir returns empty generator
            result = discover_fonts(nonexistent)
            self.assertEqual(result, [])


class TestFigletIntegration(unittest.TestCase):
    """Integration tests that use the actual figlet binary."""

    @classmethod
    def setUpClass(cls) -> None:
        """Check if figlet is available."""
        cls.figlet_available = shutil.which("figlet") is not None

    def test_figlet_available_skip_if_missing(self) -> None:
        """Skip test if figlet is not installed."""
        if not self.figlet_available:
            self.skipTest("figlet binary not found in PATH")

        # Run a simple figlet command
        result = subprocess.run(
            ["figlet", "TEST"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.strip())

    def test_figlet_output_is_valid(self) -> None:
        """Figlet output for simple word should pass is_valid_output()."""
        if not self.figlet_available:
            self.skipTest("figlet binary not found in PATH")

        result = subprocess.run(
            ["figlet", "HI"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        self.assertEqual(result.returncode, 0)

        output = result.stdout
        self.assertTrue(
            is_valid_output(output), f"FIGlet output should be valid:\n{output}"
        )

    def test_figlet_empty_input_handled(self) -> None:
        """Figlet with empty input should be handled."""
        if not self.figlet_available:
            self.skipTest("figlet binary not found in PATH")

        result = subprocess.run(
            ["figlet", ""],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        # Empty input may produce empty output or minimal output
        # The key is it shouldn't crash
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
