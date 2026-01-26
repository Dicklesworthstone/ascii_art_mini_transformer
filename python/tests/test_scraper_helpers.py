"""
Unit tests for scraper helper utilities (bd-1kv).

Tests cover:
- scrape_asciiart.py: URL normalization, same-site filtering, category parsing
- scrape_textfiles.py: base-path enforcement, tag derivation, source_id derivation
- scrape_16colors.py: URL builders, HTML extraction, SAUCE parsing
"""

from __future__ import annotations

import unittest

# ============================================================================
# scrape_asciiart.py helpers
# ============================================================================

from python.data.scrape_asciiart import (
    _extract_gallery_links,
    _normalize_base_url,
    _path_category_parts,
    _rate_limit,
    _same_site,
)


class TestAsciiArtNormalizeBaseUrl(unittest.TestCase):
    """Tests for _normalize_base_url() from scrape_asciiart.py."""

    def test_adds_trailing_slash(self) -> None:
        result = _normalize_base_url("https://example.com")
        self.assertEqual(result, "https://example.com/")

    def test_preserves_existing_trailing_slash(self) -> None:
        result = _normalize_base_url("https://example.com/")
        self.assertEqual(result, "https://example.com/")

    def test_strips_whitespace(self) -> None:
        result = _normalize_base_url("  https://example.com  ")
        self.assertEqual(result, "https://example.com/")

    def test_handles_path_in_url(self) -> None:
        result = _normalize_base_url("https://example.com/gallery")
        self.assertEqual(result, "https://example.com/gallery/")


class TestAsciiArtSameSite(unittest.TestCase):
    """Tests for _same_site() from scrape_asciiart.py."""

    def test_same_netloc_returns_true(self) -> None:
        self.assertTrue(_same_site("https://example.com/", "https://example.com/page"))

    def test_different_netloc_returns_false(self) -> None:
        self.assertFalse(_same_site("https://example.com/", "https://other.com/page"))

    def test_subdomain_is_different_site(self) -> None:
        self.assertFalse(
            _same_site("https://example.com/", "https://sub.example.com/page")
        )

    def test_different_ports_are_different_sites(self) -> None:
        self.assertFalse(
            _same_site("https://example.com:443/", "https://example.com:8080/page")
        )


class TestAsciiArtPathCategoryParts(unittest.TestCase):
    """Tests for _path_category_parts() from scrape_asciiart.py."""

    def test_gallery_path_returns_none(self) -> None:
        """The /gallery path is just the entry point, no category."""
        cat, subcat = _path_category_parts(
            "https://example.com/gallery", "https://example.com/"
        )
        self.assertIsNone(cat)
        self.assertIsNone(subcat)

    def test_category_only(self) -> None:
        cat, subcat = _path_category_parts(
            "https://example.com/animals", "https://example.com/"
        )
        self.assertEqual(cat, "animals")
        self.assertIsNone(subcat)

    def test_category_and_subcategory(self) -> None:
        cat, subcat = _path_category_parts(
            "https://example.com/animals/cats", "https://example.com/"
        )
        self.assertEqual(cat, "animals")
        self.assertEqual(subcat, "cats")

    def test_gallery_prefixed_paths_are_supported(self) -> None:
        cat, subcat = _path_category_parts(
            "https://example.com/gallery/animals/cats", "https://example.com/"
        )
        self.assertEqual(cat, "animals")
        self.assertEqual(subcat, "cats")

    def test_different_netloc_returns_none(self) -> None:
        cat, subcat = _path_category_parts(
            "https://other.com/animals", "https://example.com/"
        )
        self.assertIsNone(cat)
        self.assertIsNone(subcat)


class TestAsciiArtExtractGalleryLinks(unittest.TestCase):
    """Tests for _extract_gallery_links() from scrape_asciiart.py."""

    def test_extracts_card_gallery_links(self) -> None:
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <a class="card-gallery" href="/animals">Animals</a>
            <a class="card-gallery" href="/plants">Plants</a>
            <a class="other" href="/ignored">Ignored</a>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        base_url = "https://example.com/"
        links = _extract_gallery_links(soup, base_url)

        self.assertEqual(len(links), 2)
        self.assertIn("https://example.com/animals", links)
        self.assertIn("https://example.com/plants", links)

    def test_filters_external_links(self) -> None:
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <a class="card-gallery" href="/local">Local</a>
            <a class="card-gallery" href="https://external.com/page">External</a>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = _extract_gallery_links(soup, "https://example.com/")

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0], "https://example.com/local")

    def test_deduplicates_links(self) -> None:
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <a class="card-gallery" href="/same">Same</a>
            <a class="card-gallery" href="/same">Same Again</a>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = _extract_gallery_links(soup, "https://example.com/")

        self.assertEqual(len(links), 1)

    def test_skips_missing_or_empty_href(self) -> None:
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <a class="card-gallery">No href</a>
            <a class="card-gallery" href="">Empty</a>
            <a class="card-gallery" href="/ok">OK</a>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = _extract_gallery_links(soup, "https://example.com/")
        self.assertEqual(links, ["https://example.com/ok"])


class TestAsciiArtRateLimit(unittest.TestCase):
    def test_rate_limit_sleeps_when_delay_positive(self) -> None:
        _rate_limit(0.001, 0.0)


# ============================================================================
# scrape_textfiles.py helpers
# ============================================================================

from python.data.scrape_textfiles import (  # noqa: E402
    _filename_from_url,
    _path_category_parts as tf_path_category_parts,
    _source_id_from_url,
    _tags_for_path,
    _within_base_path,
)


class TestTextFilesWithinBasePath(unittest.TestCase):
    """Tests for _within_base_path() from scrape_textfiles.py."""

    def test_same_path_returns_true(self) -> None:
        self.assertTrue(
            _within_base_path(
                "http://example.com/ansi/", "http://example.com/ansi/file.txt"
            )
        )

    def test_subpath_returns_true(self) -> None:
        self.assertTrue(
            _within_base_path(
                "http://example.com/ansi/", "http://example.com/ansi/subdir/file.txt"
            )
        )

    def test_parent_path_returns_false(self) -> None:
        self.assertFalse(
            _within_base_path("http://example.com/ansi/", "http://example.com/file.txt")
        )

    def test_sibling_path_returns_false(self) -> None:
        self.assertFalse(
            _within_base_path(
                "http://example.com/ansi/", "http://example.com/other/file.txt"
            )
        )

    def test_different_host_returns_false(self) -> None:
        self.assertFalse(
            _within_base_path(
                "http://example.com/ansi/", "http://other.com/ansi/file.txt"
            )
        )


class TestTextFilesSourceIdFromUrl(unittest.TestCase):
    """Tests for _source_id_from_url() from scrape_textfiles.py."""

    def test_strips_base_path(self) -> None:
        result = _source_id_from_url(
            "http://example.com/ansi/subdir/file.txt", "http://example.com/ansi/"
        )
        self.assertEqual(result, "subdir/file.txt")

    def test_handles_file_in_base_path(self) -> None:
        result = _source_id_from_url(
            "http://example.com/ansi/file.txt", "http://example.com/ansi/"
        )
        self.assertEqual(result, "file.txt")

    def test_handles_different_host(self) -> None:
        result = _source_id_from_url(
            "http://other.com/path/file.txt", "http://example.com/ansi/"
        )
        # Returns the full path when hosts differ
        self.assertEqual(result, "path/file.txt")


class TestTextFilesTagsForPath(unittest.TestCase):
    """Tests for _tags_for_path() from scrape_textfiles.py."""

    def test_includes_textfiles_tag(self) -> None:
        tags = _tags_for_path(None, None, None)
        self.assertIn("textfiles", tags)

    def test_includes_category(self) -> None:
        tags = _tags_for_path("artscene", None, None)
        self.assertIn("artscene", tags)

    def test_includes_subcategory(self) -> None:
        tags = _tags_for_path("artscene", "bbs", None)
        self.assertIn("bbs", tags)

    def test_includes_file_extension(self) -> None:
        tags = _tags_for_path(None, None, "art.ans")
        self.assertIn("ans", tags)

    def test_extension_is_lowercased(self) -> None:
        tags = _tags_for_path(None, None, "art.ANS")
        self.assertIn("ans", tags)

    def test_no_extension_no_ext_tag(self) -> None:
        tags = _tags_for_path(None, None, "README")
        # Should not have empty string as tag
        self.assertNotIn("", tags)

    def test_deduplicates_tags(self) -> None:
        tags = _tags_for_path("textfiles", None, None)
        # "textfiles" appears in both base tags and category
        self.assertEqual(tags.count("textfiles"), 1)


class TestTextFilesFilenameFromUrl(unittest.TestCase):
    """Tests for _filename_from_url() from scrape_textfiles.py."""

    def test_extracts_filename(self) -> None:
        result = _filename_from_url("http://example.com/path/to/file.txt")
        self.assertEqual(result, "file.txt")

    def test_handles_no_path(self) -> None:
        result = _filename_from_url("http://example.com/")
        self.assertIsNone(result)

    def test_handles_directory_path(self) -> None:
        result = _filename_from_url("http://example.com/path/")
        self.assertIsNone(result)


class TestTextFilesPathCategoryParts(unittest.TestCase):
    """Tests for _path_category_parts() from scrape_textfiles.py."""

    def test_extracts_category_from_path(self) -> None:
        cat, subcat = tf_path_category_parts(
            "http://example.com/ansi/groups/file.txt", "http://example.com/ansi/"
        )
        self.assertEqual(cat, "groups")

    def test_extracts_subcategory_when_deep(self) -> None:
        cat, subcat = tf_path_category_parts(
            "http://example.com/ansi/groups/bbs/logo/file.txt",
            "http://example.com/ansi/",
        )
        self.assertEqual(cat, "groups")
        self.assertEqual(subcat, "bbs")

    def test_no_subcategory_when_shallow(self) -> None:
        cat, subcat = tf_path_category_parts(
            "http://example.com/ansi/groups/file.txt", "http://example.com/ansi/"
        )
        self.assertIsNone(subcat)


# ============================================================================
# scrape_16colors.py helpers
# ============================================================================

from python.data.scrape_16colors import (  # noqa: E402
    _extension,
    _extract_file_page_links,
    _extract_pack_links,
    _extract_year_links,
    _file_page_to_raw_url,
    decode_cp437,
    strip_sauce,
)


class TestSixteenColorsExtension(unittest.TestCase):
    """Tests for _extension() from scrape_16colors.py."""

    def test_extracts_extension(self) -> None:
        self.assertEqual(_extension("file.ans"), "ans")

    def test_lowercases_extension(self) -> None:
        self.assertEqual(_extension("file.ANS"), "ans")

    def test_no_extension_returns_empty(self) -> None:
        self.assertEqual(_extension("README"), "")

    def test_multiple_dots_gets_last(self) -> None:
        self.assertEqual(_extension("file.backup.txt"), "txt")


class TestSixteenColorsFilePageToRawUrl(unittest.TestCase):
    """Tests for _file_page_to_raw_url() from scrape_16colors.py."""

    def test_converts_file_page_to_raw(self) -> None:
        result = _file_page_to_raw_url(
            "https://16colo.rs/pack/mypack/file.ans", "mypack", "https://16colo.rs/"
        )
        self.assertEqual(result, "https://16colo.rs/pack/mypack/raw/file.ans")

    def test_wrong_pack_returns_none(self) -> None:
        result = _file_page_to_raw_url(
            "https://16colo.rs/pack/otherpack/file.ans",
            "mypack",
            "https://16colo.rs/",
        )
        self.assertIsNone(result)

    def test_wrong_path_structure_returns_none(self) -> None:
        result = _file_page_to_raw_url(
            "https://16colo.rs/year/2020", "mypack", "https://16colo.rs/"
        )
        self.assertIsNone(result)


class TestSixteenColorsExtractYearLinks(unittest.TestCase):
    """Tests for _extract_year_links() from scrape_16colors.py."""

    def test_extracts_year_links_from_select(self) -> None:
        html = """
        <html>
        <body>
            <select id="selecty">
                <option value="/year/2020">2020</option>
                <option value="/year/2019">2019</option>
            </select>
        </body>
        </html>
        """
        links = _extract_year_links(html, "https://16colo.rs/")

        self.assertEqual(len(links), 2)
        self.assertIn("https://16colo.rs/year/2020", links)
        self.assertIn("https://16colo.rs/year/2019", links)

    def test_returns_sorted_links(self) -> None:
        html = """
        <html>
        <body>
            <select id="selecty">
                <option value="/year/2020">2020</option>
                <option value="/year/2019">2019</option>
                <option value="/year/2021">2021</option>
            </select>
        </body>
        </html>
        """
        links = _extract_year_links(html, "https://16colo.rs/")

        # Should be sorted
        self.assertEqual(
            links,
            [
                "https://16colo.rs/year/2019",
                "https://16colo.rs/year/2020",
                "https://16colo.rs/year/2021",
            ],
        )

    def test_filters_external_links(self) -> None:
        html = """
        <html>
        <body>
            <select id="selecty">
                <option value="/year/2020">2020</option>
                <option value="https://external.com/year/2019">External</option>
            </select>
        </body>
        </html>
        """
        links = _extract_year_links(html, "https://16colo.rs/")

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0], "https://16colo.rs/year/2020")


class TestSixteenColorsExtractPackLinks(unittest.TestCase):
    """Tests for _extract_pack_links() from scrape_16colors.py."""

    def test_extracts_pack_links(self) -> None:
        html = """
        <html>
        <body>
            <a class="dizname" href="/pack/pack1">Pack 1</a>
            <a class="dizname" href="/pack/pack2">Pack 2</a>
            <a class="other" href="/pack/ignored">Ignored</a>
        </body>
        </html>
        """
        links = _extract_pack_links(html, "https://16colo.rs/")

        self.assertEqual(len(links), 2)
        self.assertIn("https://16colo.rs/pack/pack1", links)
        self.assertIn("https://16colo.rs/pack/pack2", links)

    def test_skips_non_pack_links(self) -> None:
        html = """
        <html>
        <body>
            <a class="dizname" href="/pack/pack1">Pack 1</a>
            <a class="dizname" href="/year/2020">Not a pack</a>
        </body>
        </html>
        """
        links = _extract_pack_links(html, "https://16colo.rs/")

        self.assertEqual(len(links), 1)


class TestSixteenColorsExtractFilePageLinks(unittest.TestCase):
    """Tests for _extract_file_page_links() from scrape_16colors.py."""

    def test_extracts_file_links(self) -> None:
        html = """
        <html>
        <body>
            <a href="/pack/mypack/file1.ans">File 1</a>
            <a href="/pack/mypack/file2.txt">File 2</a>
        </body>
        </html>
        """
        links = _extract_file_page_links(html, "https://16colo.rs/", "mypack")

        self.assertEqual(len(links), 2)

    def test_skips_raw_and_data_links(self) -> None:
        html = """
        <html>
        <body>
            <a href="/pack/mypack/file.ans">File</a>
            <a href="/pack/mypack/raw/file.ans">Raw</a>
            <a href="/pack/mypack/data/file.json">Data</a>
            <a href="/pack/mypack/x1/file.png">Thumbnail</a>
        </body>
        </html>
        """
        links = _extract_file_page_links(html, "https://16colo.rs/", "mypack")

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0], "https://16colo.rs/pack/mypack/file.ans")

    def test_skips_wrong_pack(self) -> None:
        html = """
        <html>
        <body>
            <a href="/pack/mypack/file.ans">My Pack File</a>
            <a href="/pack/otherpack/file.ans">Other Pack File</a>
        </body>
        </html>
        """
        links = _extract_file_page_links(html, "https://16colo.rs/", "mypack")

        self.assertEqual(len(links), 1)


class TestSixteenColorsDecodeCp437(unittest.TestCase):
    """Tests for decode_cp437() from scrape_16colors.py."""

    def test_decodes_ascii_subset(self) -> None:
        data = b"Hello World"
        result = decode_cp437(data)
        self.assertEqual(result, "Hello World")

    def test_decodes_cp437_characters(self) -> None:
        # 0xDB = █ (full block) in CP437
        data = b"\xdb\xdb\xdb"
        result = decode_cp437(data)
        self.assertEqual(result, "███")

    def test_normalizes_newlines(self) -> None:
        # CRLF -> LF
        data = b"line1\r\nline2"
        result = decode_cp437(data)
        self.assertEqual(result, "line1\nline2")


class TestSixteenColorsStripSauce(unittest.TestCase):
    """Tests for strip_sauce() from scrape_16colors.py."""

    def test_no_sauce_returns_original(self) -> None:
        data = b"Just some text without SAUCE"
        content, meta = strip_sauce(data)
        self.assertEqual(content, data)
        self.assertIsNone(meta)

    def test_short_data_returns_original(self) -> None:
        data = b"Too short"
        content, meta = strip_sauce(data)
        self.assertEqual(content, data)
        self.assertIsNone(meta)

    def test_strips_sauce_record(self) -> None:
        # Create a minimal valid SAUCE record (128 bytes starting with SAUCE00)
        sauce_record = b"SAUCE00" + b"\x00" * 121
        data = b"Content here" + sauce_record
        content, meta = strip_sauce(data)
        self.assertEqual(content, b"Content here")
        self.assertIsNotNone(meta)

    def test_extracts_sauce_title(self) -> None:
        # SAUCE title is at offset 7, length 35
        sauce_record = b"SAUCE00" + b"Test Title" + b"\x00" * 25 + b"\x00" * 86
        data = b"Content" + sauce_record
        content, meta = strip_sauce(data)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.title, "Test Title")  # type: ignore[union-attr]

    def test_strips_eof_marker(self) -> None:
        # 0x1A is often used as EOF marker before SAUCE
        sauce_record = b"SAUCE00" + b"\x00" * 121
        data = b"Content" + b"\x1a\x1a" + sauce_record
        content, meta = strip_sauce(data)
        self.assertEqual(content, b"Content")


if __name__ == "__main__":
    unittest.main()
