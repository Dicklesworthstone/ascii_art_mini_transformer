"""
Unit tests for FIGlet dataset generator helpers (bd-1nj).

Tests cover:
- _word_list() word set selection
- is_valid_output() width/height thresholds
- discover_fonts() directory traversal and sorting
- Optional integration test with actual figlet binary
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from python.data.generate_figlet import (
    COMMON_WORDS,
    EXTRA_WORDS,
    _word_list,
    discover_fonts,
    generate_dataset,
    generate_figlet,
    generate_texts_for_font,
    is_valid_output,
    main,
    retry_on_lock,
    test_font as figlet_test_font,
)


def _make_temp_dir(prefix: str) -> Path:
    # Project policy forbids auto-deleting temp dirs from tests.
    return Path(tempfile.mkdtemp(prefix=prefix))


def _install_fake_figlet(bin_dir: Path) -> Path:
    figlet = bin_dir / "figlet"
    figlet.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
font = None
if "-f" in args:
    i = args.index("-f")
    if i + 1 < len(args):
        font = args[i + 1]
    args = args[:i] + args[i + 2 :]

text = " ".join(args)
stem = pathlib.Path(font).stem if font else ""

if "fail" in stem:
    raise SystemExit(2)
if "empty" in stem:
    print("", end="")
    raise SystemExit(0)
if "wide" in stem:
    print("X" * 501)
    raise SystemExit(0)
if "tall" in stem:
    for _ in range(101):
        print("X")
    raise SystemExit(0)

print(text)
""",
        encoding="utf-8",
    )
    figlet.chmod(0o755)
    return figlet


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
        tmpdir = _make_temp_dir("ascii_figlet_fonts_empty_")
        result = discover_fonts(tmpdir)
        self.assertEqual(result, [])

    def test_finds_flf_files(self) -> None:
        """Should find .flf files in directory."""
        tmppath = _make_temp_dir("ascii_figlet_fonts_flf_")
        # Create some .flf files
        (tmppath / "font1.flf").touch()
        (tmppath / "font2.flf").touch()

        result = discover_fonts(tmppath)
        self.assertEqual(len(result), 2)

    def test_ignores_non_flf_files(self) -> None:
        """Should ignore non-.flf files."""
        tmppath = _make_temp_dir("ascii_figlet_fonts_ignore_")
        (tmppath / "font.flf").touch()
        (tmppath / "readme.txt").touch()
        (tmppath / "font.ttf").touch()

        result = discover_fonts(tmppath)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "font.flf")

    def test_finds_fonts_in_subdirectories(self) -> None:
        """Should recursively find .flf files in subdirectories."""
        tmppath = _make_temp_dir("ascii_figlet_fonts_subdirs_")
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
        tmppath = _make_temp_dir("ascii_figlet_fonts_sorted_")
        # Create fonts in non-alphabetical order
        (tmppath / "zebra.flf").touch()
        (tmppath / "alpha.flf").touch()
        (tmppath / "beta.flf").touch()

        result = discover_fonts(tmppath)
        names = [p.name for p in result]
        self.assertEqual(names, ["alpha.flf", "beta.flf", "zebra.flf"])

    def test_returns_path_objects(self) -> None:
        """Should return Path objects, not strings."""
        tmppath = _make_temp_dir("ascii_figlet_fonts_paths_")
        (tmppath / "font.flf").touch()

        result = discover_fonts(tmppath)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Path)

    def test_handles_nonexistent_directory(self) -> None:
        """Non-existent directory should return empty list (rglob behavior)."""
        tmpdir = _make_temp_dir("ascii_figlet_fonts_nonexistent_")
        nonexistent = tmpdir / "nonexistent"
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


class TestFakeFigletHelpers(unittest.TestCase):
    def test_retry_on_lock_retries_then_succeeds(self) -> None:
        attempts = 0

        def _flaky() -> int:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise sqlite3.OperationalError("database is locked")
            return 123

        out = retry_on_lock(_flaky, max_retries=5, base_delay=0.0, max_delay=0.0)
        self.assertEqual(out, 123)
        self.assertEqual(attempts, 3)

    def test_retry_on_lock_raises_after_exhaustion(self) -> None:
        def _always_locked() -> int:
            raise sqlite3.OperationalError("database is locked")

        with self.assertRaises(sqlite3.OperationalError):
            retry_on_lock(_always_locked, max_retries=2, base_delay=0.0, max_delay=0.0)

    def test_test_font_and_generate_figlet_with_fake_binary(self) -> None:
        tmpdir = _make_temp_dir("ascii_fake_figlet_")
        bin_dir = tmpdir / "bin"
        bin_dir.mkdir()
        _install_fake_figlet(bin_dir)

        fonts_dir = tmpdir / "fonts"
        fonts_dir.mkdir()
        good = fonts_dir / "good.flf"
        good.touch()
        wide = fonts_dir / "wide.flf"
        wide.touch()
        tall = fonts_dir / "tall.flf"
        tall.touch()
        empty = fonts_dir / "empty.flf"
        empty.touch()
        fail = fonts_dir / "fail.flf"
        fail.touch()

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
        try:
            ok = figlet_test_font(good)
            self.assertTrue(ok.works)

            too_wide = figlet_test_font(wide)
            self.assertFalse(too_wide.works)
            self.assertIn("too wide", too_wide.reason.lower())

            too_tall = figlet_test_font(tall)
            self.assertFalse(too_tall.works)
            self.assertIn("too tall", too_tall.reason.lower())

            empty_out = figlet_test_font(empty)
            self.assertFalse(empty_out.works)
            self.assertIn("empty", empty_out.reason.lower())

            failed = figlet_test_font(fail)
            self.assertFalse(failed.works)
            self.assertIn("exit code", failed.reason.lower())

            self.assertIsNone(generate_figlet("HI", fail))
            out = generate_figlet("HI", good)
            self.assertIsNotNone(out)
            self.assertTrue(out.strip())
        finally:
            os.environ["PATH"] = old_path

    def test_generate_texts_for_font_emits_letter_digit_and_word(self) -> None:
        tmpdir = _make_temp_dir("ascii_fake_figlet_texts_")
        bin_dir = tmpdir / "bin"
        bin_dir.mkdir()
        _install_fake_figlet(bin_dir)

        fonts_dir = tmpdir / "fonts"
        fonts_dir.mkdir()
        font_path = fonts_dir / "good.flf"
        font_path.touch()

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
        try:
            letter_art, _desc, meta = next(
                generate_texts_for_font(
                    font_path,
                    "good",
                    include_letters=True,
                    include_digits=False,
                    words=[],
                )
            )
            self.assertEqual(meta["type"], "letter")
            self.assertTrue(letter_art.strip())

            digit_art, _desc, meta = next(
                generate_texts_for_font(
                    font_path,
                    "good",
                    include_letters=False,
                    include_digits=True,
                    words=[],
                )
            )
            self.assertEqual(meta["type"], "digit")
            self.assertTrue(digit_art.strip())

            word_art, _desc, meta = next(
                generate_texts_for_font(
                    font_path,
                    "good",
                    include_letters=False,
                    include_digits=False,
                    words=["HI"],
                )
            )
            self.assertEqual(meta["type"], "word")
            self.assertEqual(meta["input_text"], "HI")
            self.assertTrue(word_art.strip())
        finally:
            os.environ["PATH"] = old_path

    def test_generate_dataset_inserts_rows_with_fake_figlet(self) -> None:
        tmpdir = _make_temp_dir("ascii_fake_figlet_dataset_")
        bin_dir = tmpdir / "bin"
        bin_dir.mkdir()
        _install_fake_figlet(bin_dir)

        fonts_dir = tmpdir / "fonts"
        fonts_dir.mkdir()
        (fonts_dir / "good.flf").touch()

        db_path = tmpdir / "figlet.db"

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
        try:
            stats = generate_dataset(
                fonts_dir=fonts_dir,
                db_path=str(db_path),
                batch_size=1,
                max_fonts=1,
                include_letters=False,
                include_digits=False,
                word_set="base",
                stop_at_total_rows=1,
            )
        finally:
            os.environ["PATH"] = old_path

        self.assertEqual(stats.fonts_tested, 1)
        self.assertEqual(stats.fonts_working, 1)
        self.assertGreaterEqual(stats.total_generated, 1)
        self.assertGreaterEqual(stats.inserted, 1)
        self.assertEqual(stats.errors, 0)

        conn = sqlite3.connect(str(db_path))
        try:
            total = conn.execute("SELECT COUNT(*) FROM ascii_art").fetchone()[0]
        finally:
            conn.close()
        self.assertGreaterEqual(int(total), 1)

    def test_main_runs_with_fake_figlet(self) -> None:
        tmpdir = _make_temp_dir("ascii_fake_figlet_main_")
        bin_dir = tmpdir / "bin"
        bin_dir.mkdir()
        _install_fake_figlet(bin_dir)

        fonts_dir = tmpdir / "fonts"
        fonts_dir.mkdir()
        (fonts_dir / "good.flf").touch()

        db_path = tmpdir / "main.db"

        old_path = os.environ.get("PATH", "")
        old_argv = list(sys.argv)
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
        sys.argv = [
            "generate_figlet",
            "--fonts-dir",
            str(fonts_dir),
            "--db-path",
            str(db_path),
            "--batch-size",
            "1",
            "--max-fonts",
            "1",
            "--mode",
            "words",
            "--stop-at-total-rows",
            "1",
        ]
        try:
            self.assertEqual(main(), 0)
        finally:
            os.environ["PATH"] = old_path
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
