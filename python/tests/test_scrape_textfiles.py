from __future__ import annotations

from python.data.scrape_textfiles import _extract_listing_hrefs


def test_extract_listing_hrefs_filters_parent_and_dedupes() -> None:
    html = """
    <html>
      <body>
        <pre>
          <a href="../">Parent Directory</a>
          <a href="?C=N;O=D">Name</a>
          <a href="mailto:test@example.com">mail</a>
          <a href="javascript:alert(1)">js</a>
          <a href="bbs/">bbs/</a>
          <a href="scene/">scene/</a>
          <a href="bbs/">bbs/</a>
          <a href="logo.ans">logo.ans</a>
          <a href="file.txt">file.txt</a>
          <a href="#ignore">ignore</a>
        </pre>
      </body>
    </html>
    """

    hrefs = _extract_listing_hrefs(html)
    assert hrefs == ["bbs/", "scene/", "logo.ans", "file.txt"]
