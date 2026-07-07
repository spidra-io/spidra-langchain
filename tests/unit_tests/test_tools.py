"""Unit tests for the Spidra tools (mocked client, no network)."""

from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from langchain_spidra import SpidraBatchScrape, SpidraCrawl, SpidraScrape


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


@dataclass
class _FakeCrawlJob:
    result: List[_FakeCrawlPage]
    status: str = "completed"


@dataclass
class _FakeBatchItem:
    url: str
    result: Any
    status: str = "completed"


@dataclass
class _FakeBatchResponse:
    items: List[_FakeBatchItem]
    status: str = "completed"


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
    """Crawl returns one dict per page in ``job.result``."""
    tool = SpidraCrawl(api_key="spd-test")  # type: ignore[arg-type]
    tool.client = MagicMock()
    tool.client.crawl.return_value = _FakeCrawlJob(
        result=[
            _FakeCrawlPage("https://example.com/a", "Page A", "A"),
            _FakeCrawlPage("https://example.com/b", "Page B", None),
        ]
    )

    result = tool.invoke({"url": "https://example.com", "max_pages": 10})

    assert [p["url"] for p in result] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert result[0]["title"] == "A"
    assert result[0]["data"] == "Page A"
    kwargs = tool.client.crawl.call_args.kwargs
    assert kwargs["max_pages"] == 10


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


@pytest.mark.parametrize(
    ("tool_cls", "method", "params"),
    [
        (SpidraScrape, "scrape", {"url": "https://example.com"}),
        (SpidraCrawl, "crawl", {"url": "https://example.com"}),
        (SpidraBatchScrape, "batch_scrape", {"urls": ["https://example.com"]}),
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
