"""Unit tests for ``SpidraLoader`` — all SDK calls are mocked, no network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_spidra.document_loaders import (
    SpidraLoader,
    SpidraMapLoader,
    SpidraParseLoader,
    SpidraSearchLoader,
    _split_multi_url_content,
    _to_str,
)

# ---------------------------------------------------------------------------
# Fakes matching SDK response shapes
# ---------------------------------------------------------------------------


@dataclass
class _FakeScrapeResult:
    content: Any
    ai_extraction_failed: bool = False
    extraction_empty: bool = False


@dataclass
class _FakeCrawlPage:
    url: str
    data: Any
    title: str | None = None
    depth: int | None = None
    parent_url: str | None = None


@dataclass
class _FakeCrawlJob:
    result: list[_FakeCrawlPage]
    status: str = "completed"
    job_id: str = "crawl-123"
    result_truncated: bool = False


@dataclass
class _FakeSearchResultItem:
    title: str
    url: str
    position: int = 1
    description: str | None = None
    markdown: str | None = None
    source: str | None = None
    date: str | None = None
    authors: str | None = None
    doi: str | None = None


@dataclass
class _FakeSearchData:
    web: list[_FakeSearchResultItem] | None = None
    news: list[_FakeSearchResultItem] | None = None
    research: list[_FakeSearchResultItem] | None = None
    developer: list[_FakeSearchResultItem] | None = None
    images: list[_FakeSearchResultItem] | None = None
    videos: list[_FakeSearchResultItem] | None = None
    extras: Any = None


@dataclass
class _FakeSearchResult:
    data: _FakeSearchData


@dataclass
class _FakeMapLink:
    url: str
    title: str | None = None


@dataclass
class _FakeMapResult:
    links: list[_FakeMapLink]
    total: int = 1


@dataclass
class _FakeParsedPage:
    page_number: int
    markdown: str
    method: str = "text"


@dataclass
class _FakeParseJob:
    job_id: str = "parse-123"
    status: str = "completed"
    markdown: str | None = "# Doc Content"
    total_pages: int | None = 1
    pages: list[_FakeParsedPage] | None = None
    json: Any = None


@dataclass
class _FakeBatchItem:
    url: str
    result: Any
    status: str = "completed"


@dataclass
class _FakeBatchResponse:
    items: list[_FakeBatchItem]
    status: str = "completed"


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the loader's client factory to return a mock client."""
    client = MagicMock()
    monkeypatch.setattr(
        "langchain_spidra.document_loaders.create_client",
        lambda **_: client,
    )
    return client


# ---------------------------------------------------------------------------
# _to_str utility
# ---------------------------------------------------------------------------


def test_to_str_none_returns_empty() -> None:
    assert _to_str(None) == ""


def test_to_str_str_passthrough() -> None:
    assert _to_str("hello") == "hello"


def test_to_str_dict_json_serialised() -> None:
    result = _to_str({"key": "value"})
    assert '"key"' in result
    assert '"value"' in result


def test_to_str_list_json_serialised() -> None:
    assert _to_str([1, 2, 3]) == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# _split_multi_url_content utility
# ---------------------------------------------------------------------------


def test_split_single_section_with_prefix() -> None:
    content = "Content from: https://example.com\n# Hello World\nSome text."
    pairs = _split_multi_url_content(content, ["https://example.com"])
    assert len(pairs) == 1
    assert pairs[0][0] == "https://example.com"
    assert "Hello World" in pairs[0][1]


def test_split_two_sections() -> None:
    content = (
        "Content from: https://a.com\nPage A content\n\n---\n\n"
        "Content from: https://b.com\nPage B content"
    )
    pairs = _split_multi_url_content(content, ["https://a.com", "https://b.com"])
    assert len(pairs) == 2
    assert pairs[0][0] == "https://a.com"
    assert pairs[1][0] == "https://b.com"


def test_split_fallback_url_when_no_prefix() -> None:
    pairs = _split_multi_url_content("Some content", ["https://fallback.com"])
    assert len(pairs) == 1
    assert pairs[0][0] == "https://fallback.com"


def test_split_empty_content_returns_empty_list() -> None:
    assert _split_multi_url_content("", []) == []


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loader raises ``ValueError`` when no API key is configured."""
    monkeypatch.delenv("SPIDRA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        SpidraLoader("https://example.com")


def test_raises_without_url_for_scrape(mock_client: MagicMock) -> None:
    with pytest.raises(ValueError, match="'url' is required"):
        SpidraLoader(api_key="spd_test")


def test_raises_without_urls_for_batch(mock_client: MagicMock) -> None:
    with pytest.raises(ValueError, match="'urls' is required"):
        SpidraLoader(mode="batch", api_key="spd_test")


def test_url_wrapped_into_urls_list(mock_client: MagicMock) -> None:
    loader = SpidraLoader("https://example.com", api_key="spd_test")
    assert loader._urls == ["https://example.com"]


def test_explicit_urls_list(mock_client: MagicMock) -> None:
    loader = SpidraLoader(
        urls=["https://a.com", "https://b.com"],
        mode="batch",
        api_key="spd_test",
    )
    assert len(loader._urls) == 2


# ---------------------------------------------------------------------------
# Scrape mode
# ---------------------------------------------------------------------------


def test_scrape_raw_markdown_one_doc(mock_client: MagicMock) -> None:
    """Raw markdown scrape yields one document per URL section."""
    mock_client.scrape.return_value = _FakeScrapeResult(
        content="Content from: https://example.com\n# Title\nBody text."
    )
    loader = SpidraLoader("https://example.com", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "https://example.com"
    assert "Body text" in docs[0].page_content


def test_scrape_two_urls_yields_two_docs(mock_client: MagicMock) -> None:
    mock_client.scrape.return_value = _FakeScrapeResult(
        content=(
            "Content from: https://a.com\nA content\n\n---\n\n"
            "Content from: https://b.com\nB content"
        )
    )
    loader = SpidraLoader(urls=["https://a.com", "https://b.com"], api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "https://a.com"
    assert docs[1].metadata["source"] == "https://b.com"


def test_scrape_prompt_yields_single_ai_doc(mock_client: MagicMock) -> None:
    """Scrape with a prompt yields one document with AI-extracted content."""
    mock_client.scrape.return_value = _FakeScrapeResult(content={"title": "Hello"})
    loader = SpidraLoader(
        "https://example.com", api_key="spd_test", params={"prompt": "Extract the title"}
    )
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert '"title"' in docs[0].page_content
    assert docs[0].metadata["ai_extraction_failed"] is False


def test_scrape_at_most_3_urls(mock_client: MagicMock) -> None:
    """Only up to 3 URLs are sent in a single scrape call."""
    mock_client.scrape.return_value = _FakeScrapeResult(content="")
    loader = SpidraLoader(
        urls=["https://a.com", "https://b.com", "https://c.com", "https://d.com"],
        api_key="spd_test",
    )
    list(loader.lazy_load())
    scrape_params = mock_client.scrape.call_args[0][0]
    assert len(scrape_params.urls) == 3


def test_scrape_none_content_fallback(mock_client: MagicMock) -> None:
    mock_client.scrape.return_value = _FakeScrapeResult(content=None)
    loader = SpidraLoader("https://example.com", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].page_content == ""


# ---------------------------------------------------------------------------
# Crawl mode
# ---------------------------------------------------------------------------


def test_crawl_yields_one_doc_per_page(mock_client: MagicMock) -> None:
    mock_client.crawl.return_value = _FakeCrawlJob(
        result=[
            _FakeCrawlPage("https://docs.example.com/intro", "Intro page", "Introduction"),
            _FakeCrawlPage("https://docs.example.com/api", "API reference", "API"),
        ]
    )
    loader = SpidraLoader("https://docs.example.com", mode="crawl", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "https://docs.example.com/intro"
    assert docs[0].metadata["title"] == "Introduction"


def test_crawl_page_without_title(mock_client: MagicMock) -> None:
    mock_client.crawl.return_value = _FakeCrawlJob(
        result=[_FakeCrawlPage("https://example.com/page", "Content", None)]
    )
    loader = SpidraLoader("https://example.com", mode="crawl", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert "title" not in docs[0].metadata


def test_crawl_empty_data_page_skipped(mock_client: MagicMock) -> None:
    mock_client.crawl.return_value = _FakeCrawlJob(
        result=[
            _FakeCrawlPage("https://example.com/empty", None),
            _FakeCrawlPage("https://example.com/real", "Real content"),
        ]
    )
    loader = SpidraLoader("https://example.com", mode="crawl", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "https://example.com/real"


def test_crawl_custom_instruction_forwarded(mock_client: MagicMock) -> None:
    mock_client.crawl.return_value = _FakeCrawlJob(result=[])
    loader = SpidraLoader(
        "https://example.com",
        mode="crawl",
        api_key="spd_test",
        params={"crawl_instruction": "Find blog posts", "transform_instruction": "Get body"},
    )
    list(loader.lazy_load())
    crawl_params = mock_client.crawl.call_args[0][0]
    assert crawl_params.crawl_instruction == "Find blog posts"
    assert crawl_params.transform_instruction == "Get body"


def test_crawl_no_transform_instruction_defaults_to_none(mock_client: MagicMock) -> None:
    """When no transform_instruction is given, it defaults to None (no AI charge)."""
    mock_client.crawl.return_value = _FakeCrawlJob(result=[])
    loader = SpidraLoader("https://example.com", mode="crawl", api_key="spd_test")
    list(loader.lazy_load())
    crawl_params = mock_client.crawl.call_args[0][0]
    assert crawl_params.transform_instruction is None


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def test_batch_yields_one_doc_per_completed_item(mock_client: MagicMock) -> None:
    mock_client.batch_scrape.return_value = _FakeBatchResponse(
        items=[
            _FakeBatchItem("https://a.com", "Content A"),
            _FakeBatchItem("https://b.com", "Content B"),
        ]
    )
    loader = SpidraLoader(urls=["https://a.com", "https://b.com"], mode="batch", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert docs[0].page_content == "Content A"
    assert docs[1].metadata["source"] == "https://b.com"


def test_batch_failed_items_skipped(mock_client: MagicMock) -> None:
    mock_client.batch_scrape.return_value = _FakeBatchResponse(
        items=[
            _FakeBatchItem("https://a.com", "Good content"),
            _FakeBatchItem("https://b.com", None, status="failed"),
        ]
    )
    loader = SpidraLoader(urls=["https://a.com", "https://b.com"], mode="batch", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "https://a.com"


def test_batch_pending_items_skipped(mock_client: MagicMock) -> None:
    mock_client.batch_scrape.return_value = _FakeBatchResponse(
        items=[_FakeBatchItem("https://a.com", None, status="pending")]
    )
    loader = SpidraLoader(urls=["https://a.com"], mode="batch", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 0


def test_batch_load_returns_list(mock_client: MagicMock) -> None:
    mock_client.batch_scrape.return_value = _FakeBatchResponse(
        items=[_FakeBatchItem("https://a.com", "Content")]
    )
    loader = SpidraLoader(urls=["https://a.com"], mode="batch", api_key="spd_test")
    assert isinstance(loader.load(), list)


# ---------------------------------------------------------------------------
# Crawl metadata & pagination
# ---------------------------------------------------------------------------


def test_crawl_metadata_depth_and_parent_url(mock_client: MagicMock) -> None:
    mock_client.crawl.return_value = _FakeCrawlJob(
        result=[
            _FakeCrawlPage(
                "https://example.com/sub",
                "Sub page",
                "Sub",
                depth=2,
                parent_url="https://example.com",
            )
        ]
    )
    loader = SpidraLoader("https://example.com", mode="crawl", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].metadata["depth"] == 2
    assert docs[0].metadata["parent_url"] == "https://example.com"


# ---------------------------------------------------------------------------
# Search mode
# ---------------------------------------------------------------------------


def test_search_mode_yields_documents(mock_client: MagicMock) -> None:
    mock_client.search.return_value = _FakeSearchResult(
        data=_FakeSearchData(
            web=[
                _FakeSearchResultItem(
                    title="Doc A",
                    url="https://example.com/a",
                    description="Snippet A",
                )
            ]
        )
    )
    loader = SpidraLoader(query="artificial intelligence", mode="search", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].page_content == "Snippet A"
    assert docs[0].metadata["source"] == "https://example.com/a"
    assert docs[0].metadata["title"] == "Doc A"


def test_search_loader_convenience_class(mock_client: MagicMock) -> None:
    mock_client.search.return_value = _FakeSearchResult(
        data=_FakeSearchData(
            web=[
                _FakeSearchResultItem(
                    title="Doc B",
                    url="https://example.com/b",
                    markdown="# Markdown B",
                )
            ]
        )
    )
    loader = SpidraSearchLoader("AI agents", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].page_content == "# Markdown B"


def test_search_requires_query() -> None:
    with pytest.raises(ValueError, match="'query' is required"):
        SpidraLoader(mode="search", api_key="spd_test")


# ---------------------------------------------------------------------------
# Map mode
# ---------------------------------------------------------------------------


def test_map_mode_yields_documents(mock_client: MagicMock) -> None:
    mock_client.map.return_value = _FakeMapResult(
        links=[
            _FakeMapLink(url="https://docs.example.com/1", title="Page 1"),
            _FakeMapLink(url="https://docs.example.com/2", title="Page 2"),
        ]
    )
    loader = SpidraLoader("https://docs.example.com", mode="map", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert docs[0].page_content == "https://docs.example.com/1"
    assert docs[0].metadata["title"] == "Page 1"


def test_map_loader_convenience_class(mock_client: MagicMock) -> None:
    mock_client.map.return_value = _FakeMapResult(
        links=[_FakeMapLink(url="https://docs.example.com/a")]
    )
    loader = SpidraMapLoader("https://docs.example.com", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].page_content == "https://docs.example.com/a"


def test_map_requires_url() -> None:
    with pytest.raises(ValueError, match="'url' is required"):
        SpidraLoader(mode="map", api_key="spd_test")


# ---------------------------------------------------------------------------
# Parse mode
# ---------------------------------------------------------------------------


def test_parse_mode_yields_pages(mock_client: MagicMock) -> None:
    mock_client.parse.return_value = _FakeParseJob(
        pages=[
            _FakeParsedPage(page_number=1, markdown="# Page 1"),
            _FakeParsedPage(page_number=2, markdown="# Page 2"),
        ]
    )
    loader = SpidraLoader(source="https://example.com/doc.pdf", mode="parse", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert docs[0].page_content == "# Page 1"
    assert docs[0].metadata["page_number"] == 1
    assert docs[1].page_content == "# Page 2"
    assert docs[1].metadata["page_number"] == 2


def test_parse_loader_convenience_class(mock_client: MagicMock) -> None:
    mock_client.parse.return_value = _FakeParseJob(
        markdown="# Full Document Content",
        pages=None,
    )
    loader = SpidraParseLoader("document.pdf", api_key="spd_test")
    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert docs[0].page_content == "# Full Document Content"


def test_parse_requires_source() -> None:
    with pytest.raises(ValueError, match="'source' is required"):
        SpidraLoader(mode="parse", api_key="spd_test")

