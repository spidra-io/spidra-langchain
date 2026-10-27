"""Unit tests for the Spidra tools (mocked client, no network)."""

from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_spidra import (
    SpidraBatchScrape,
    SpidraBulkScrape,
    SpidraCrawl,
    SpidraMap,
    SpidraMonitor,
    SpidraParse,
    SpidraRunPreset,
    SpidraRunTemplate,
    SpidraScrape,
    SpidraSearch,
)


@dataclass
class _FakeScrapeResult:
    content: Any
    ai_extraction_failed: bool = False
    extraction_empty: bool = False


@dataclass
class _FakeCrawlPage:
    url: str
    data: Any
    title: Optional[str] = None
    depth: Optional[int] = None
    parent_url: Optional[str] = None


@dataclass
class _FakeCrawlJob:
    result: list[_FakeCrawlPage]
    status: str = "completed"


@dataclass
class _FakeBatchItem:
    url: str
    result: Any
    status: str = "completed"


@dataclass
class _FakeBatchResponse:
    items: list[_FakeBatchItem]
    status: str = "completed"


@dataclass
class _FakeSearchResultItem:
    title: str
    url: str
    position: int = 1
    description: Optional[str] = None
    markdown: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    authors: Optional[str] = None
    doi: Optional[str] = None


@dataclass
class _FakeSearchData:
    web: Optional[list[_FakeSearchResultItem]] = None
    news: Optional[list[_FakeSearchResultItem]] = None
    research: Optional[list[_FakeSearchResultItem]] = None
    developer: Optional[list[_FakeSearchResultItem]] = None
    images: Optional[list[_FakeSearchResultItem]] = None
    videos: Optional[list[_FakeSearchResultItem]] = None
    extras: Any = None


@dataclass
class _FakeSearchResult:
    data: _FakeSearchData


@dataclass
class _FakeMapLink:
    url: str
    title: Optional[str] = None


@dataclass
class _FakeMapResult:
    links: list[_FakeMapLink]
    total: int = 1
    sources: dict[str, int] = field(default_factory=lambda: {"sitemap": 1})
    credits_used: int = 1


@dataclass
class _FakeParsedPage:
    page_number: int
    markdown: str
    method: str = "text"


@dataclass
class _FakeParseJob:
    job_id: str = "parse-123"
    status: str = "completed"
    markdown: Optional[str] = "# Doc Content"
    total_pages: Optional[int] = 1
    pages: Optional[list[_FakeParsedPage]] = None
    json: Any = None


@dataclass
class _FakeBulkStats:
    pages: int = 2
    failed_pages: int = 0
    raw_items: int = 2
    unique_items: int = 2
    duplicates_removed: int = 0
    credits_used: int = 2


@dataclass
class _FakeBulkResult:
    items: list[dict[str, Any]]
    stats: _FakeBulkStats


def test_scrape_tool_returns_content() -> None:
    """Scrape returns content dict with markdown string."""
    tool = SpidraScrape(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.scrape.return_value = _FakeScrapeResult(
        content="# Hello World", ai_extraction_failed=False, extraction_empty=False
    )

    result = tool.invoke({"url": "https://example.com"})

    assert result == {
        "content": "# Hello World",
        "ai_extraction_failed": False,
        "extraction_empty": False,
    }
    call = tool.client.scrape.call_args
    assert call.args[0] == "https://example.com"


@pytest.mark.asyncio
async def test_scrape_tool_arun() -> None:
    """Async scrape tool execution."""
    tool = SpidraScrape(api_key="spd-test")  # type: ignore[arg-type]
    tool.async_client = MagicMock()
    tool.async_client.scrape = AsyncMock(
        return_value=_FakeScrapeResult(content="# Async Scrape")
    )

    result = await tool.ainvoke({"url": "https://example.com"})
    assert result["content"] == "# Async Scrape"


def test_scrape_tool_forwards_options() -> None:
    """Explicit options are forwarded to the client."""
    tool = SpidraScrape(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.scrape.return_value = _FakeScrapeResult(content="x")

    tool.invoke({"url": "https://example.com", "prompt": "Extract the title", "output": "json"})

    kwargs = tool.client.scrape.call_args.kwargs
    assert kwargs["prompt"] == "Extract the title"
    assert kwargs["output"] == "json"


def test_scrape_tool_drops_none_params() -> None:
    """None-valued optional params are not forwarded to the client."""
    tool = SpidraScrape(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.scrape.return_value = _FakeScrapeResult(content="x")

    tool.invoke({"url": "https://example.com"})

    kwargs = tool.client.scrape.call_args.kwargs
    assert "prompt" not in kwargs
    assert "output" not in kwargs
    assert "use_proxy" not in kwargs


def test_scrape_tool_serialises_dict_content() -> None:
    """Dict content from AI extraction is JSON-serialised."""
    tool = SpidraScrape(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.scrape.return_value = _FakeScrapeResult(content={"title": "Hello"})

    result = tool.invoke({"url": "https://example.com", "prompt": "Get title"})

    assert isinstance(result["content"], str)
    assert '"title"' in result["content"]


def test_crawl_tool_returns_page_list() -> None:
    """Crawl returns a list of page dicts."""
    tool = SpidraCrawl(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.crawl.return_value = _FakeCrawlJob(
        result=[_FakeCrawlPage("https://example.com/a", "Page A", "A", depth=1)]
    )

    result = tool.invoke({"url": "https://example.com", "max_pages": 10, "respect_robots": True})

    assert len(result) == 1
    assert result[0]["url"] == "https://example.com/a"
    assert result[0]["title"] == "A"
    assert result[0]["data"] == "Page A"
    assert result[0]["depth"] == 1
    kwargs = tool.client.crawl.call_args.kwargs
    assert kwargs["max_pages"] == 10
    assert kwargs["respect_robots"] is True


def test_crawl_tool_drops_none_params() -> None:
    """None-valued optional crawl params are not forwarded."""
    tool = SpidraCrawl(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.crawl.return_value = _FakeCrawlJob(result=[])

    tool.invoke({"url": "https://example.com"})

    kwargs = tool.client.crawl.call_args.kwargs
    assert "crawl_instruction" not in kwargs
    assert "transform_instruction" not in kwargs


def test_batch_scrape_tool_returns_item_list() -> None:
    """BatchScrape returns one dict per item."""
    tool = SpidraBatchScrape(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.batch_scrape.return_value = _FakeBatchResponse(
        items=[
            _FakeBatchItem("https://a.com", "Content A"),
            _FakeBatchItem("https://b.com", "Content B"),
        ]
    )

    result = tool.invoke({"urls": ["https://a.com", "https://b.com"]})

    assert [item["url"] for item in result] == ["https://a.com", "https://b.com"]
    assert result[0]["result"] == "Content A"
    assert result[0]["status"] == "completed"
    call = tool.client.batch_scrape.call_args
    assert call.args[0] == ["https://a.com", "https://b.com"]


def test_search_tool_returns_results() -> None:
    """Search tool returns structured results."""
    tool = SpidraSearch(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.search.return_value = _FakeSearchResult(
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

    result = tool.invoke({"query": "AI research", "limit": 5})

    assert result["query"] == "AI research"
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Doc A"
    assert result["results"][0]["url"] == "https://example.com/a"
    assert result["results"][0]["content"] == "Snippet A"


def test_map_tool_returns_links() -> None:
    """Map tool returns discovered links."""
    tool = SpidraMap(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.map.return_value = _FakeMapResult(
        links=[_FakeMapLink(url="https://docs.example.com/guide", title="Guide")],
        total=1,
    )

    result = tool.invoke({"url": "https://docs.example.com", "limit": 50})

    assert result["total"] == 1
    assert len(result["links"]) == 1
    assert result["links"][0]["url"] == "https://docs.example.com/guide"


def test_parse_tool_returns_pages() -> None:
    """Parse tool returns parsed pages."""
    tool = SpidraParse(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.parse.return_value = _FakeParseJob(
        markdown="# Full Doc",
        pages=[_FakeParsedPage(page_number=1, markdown="# Page 1")],
    )

    result = tool.invoke({"source": "https://example.com/doc.pdf"})

    assert result["markdown"] == "# Full Doc"
    assert len(result["pages"]) == 1
    assert result["pages"][0]["markdown"] == "# Page 1"


def test_bulk_scrape_tool_returns_items() -> None:
    """BulkScrape tool returns items and stats."""
    tool = SpidraBulkScrape(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.bulk_scrape.return_value = _FakeBulkResult(
        items=[{"id": "1", "name": "Item 1"}],
        stats=_FakeBulkStats(),
    )

    result = tool.invoke(
        {
            "template": "https://example.com/{id}",
            "variables": {"id": ["1"]},
            "items_path": "items",
            "dedupe_by": "id",
        }
    )

    assert len(result["items"]) == 1
    assert result["items"][0]["id"] == "1"
    assert result["stats"]["pages"] == 2


@dataclass
class _FakeTemplateRun:
    run_id: str = "run_123"
    status: str = "completed"
    template: str = "google-maps-search"
    tool: str = "template"
    row_count: int = 1
    url_count: int = 1
    credits_used: int = 2
    rows: list[dict[str, Any]] = field(default_factory=lambda: [{"title": "Shop 1"}])
    failures: list[Any] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class _FakePresetRun:
    preset_id: str = "pre_123"
    tool: str = "template"
    run: Optional[_FakeTemplateRun] = None
    result: Any = None


@dataclass
class _FakeMonitor:
    id: str = "mon_123"
    name: str = "Price Tracker"
    status: str = "active"
    last_check_at: Optional[str] = "2026-10-27T10:00:00Z"
    last_change_at: Optional[str] = "2026-10-27T10:00:00Z"
    last_summary: Optional[str] = "Price dropped by 10%"


@dataclass
class _FakeMonitorCheck:
    id: str = "chk_123"
    trigger: str = "schedule"
    status: str = "completed"
    has_changes: bool = True
    summary: str = "Price dropped"
    credits_used: int = 1
    started_at: str = "2026-10-27T10:00:00Z"
    finished_at: str = "2026-10-27T10:00:05Z"


@dataclass
class _FakeMonitorCheckList:
    checks: list[_FakeMonitorCheck] = field(default_factory=list)


def test_run_template_tool_returns_rows() -> None:
    """RunTemplate tool returns extracted rows and metadata."""
    tool = SpidraRunTemplate(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.run_template.return_value = _FakeTemplateRun()

    result = tool.invoke({"slug": "google-maps-search", "inputs": {"query": "dentist"}})

    assert result["status"] == "completed"
    assert result["row_count"] == 1
    assert result["rows"] == [{"title": "Shop 1"}]
    call = tool.client.run_template.call_args
    assert call.args[0] == "google-maps-search"
    assert call.args[1] == {"query": "dentist"}


def test_run_preset_tool_returns_result() -> None:
    """RunPreset tool returns executed preset run."""
    tool = SpidraRunPreset(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.run_preset.return_value = _FakePresetRun(run=_FakeTemplateRun())

    result = tool.invoke({"preset_id": "pre_123"})

    assert result["preset_id"] == "pre_123"
    assert result["run"]["run_id"] == "run_123"
    call = tool.client.run_preset.call_args
    assert call.args[0] == "pre_123"


def test_monitor_tool_returns_latest_check() -> None:
    """Monitor tool returns monitor details and latest check."""
    tool = SpidraMonitor(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.get_monitor.return_value = _FakeMonitor()
    tool.client.list_monitor_checks.return_value = _FakeMonitorCheckList(
        checks=[_FakeMonitorCheck()]
    )

    result = tool.invoke({"monitor_id": "mon_123"})

    assert result["monitor_id"] == "mon_123"
    assert result["name"] == "Price Tracker"
    assert result["latest_check"]["has_changes"] is True
    assert result["latest_check"]["summary"] == "Price dropped"


@pytest.mark.parametrize(
    ("tool_cls", "method", "params"),
    [
        (SpidraScrape, "scrape", {"url": "https://example.com"}),
        (SpidraCrawl, "crawl", {"url": "https://example.com"}),
        (SpidraBatchScrape, "batch_scrape", {"urls": ["https://example.com"]}),
        (SpidraSearch, "search", {"query": "test"}),
        (SpidraMap, "map", {"url": "https://example.com"}),
        (SpidraParse, "parse", {"source": "https://example.com/doc.pdf"}),
        (
            SpidraBulkScrape,
            "bulk_scrape",
            {
                "template": "https://example.com/{id}",
                "variables": {"id": ["1"]},
                "items_path": "items",
                "dedupe_by": "id",
            },
        ),
        (
            SpidraRunTemplate,
            "run_template",
            {"slug": "google-maps-search", "inputs": {"query": "dentist"}},
        ),
        (
            SpidraRunPreset,
            "run_preset",
            {"preset_id": "pre_123"},
        ),
        (
            SpidraMonitor,
            "get_monitor",
            {"monitor_id": "mon_123"},
        ),
    ],
)
def test_tool_returns_error_string_on_exception(
    tool_cls: type, method: str, params: dict[str, Any]
) -> None:
    """Tools return a string (not raise) when the client errors."""
    tool = tool_cls(api_key="spd-test")
    tool.client = MagicMock()
    getattr(tool.client, method).side_effect = RuntimeError("boom")

    result = tool.invoke(params)

    assert isinstance(result, str)
    assert "boom" in result
