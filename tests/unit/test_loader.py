"""
Unit tests for SpidraLoader — all SDK calls are mocked so no real HTTP occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from langchain_spidra.document_loaders import SpidraLoader, _split_multi_url_content, _to_str


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes matching the SDK response shapes
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


# ---------------------------------------------------------------------------
# _to_str utility
# ---------------------------------------------------------------------------


class TestToStr:
    def test_none_returns_empty(self) -> None:
        assert _to_str(None) == ""

    def test_str_passthrough(self) -> None:
        assert _to_str("hello") == "hello"

    def test_dict_json_serialised(self) -> None:
        result = _to_str({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_list_json_serialised(self) -> None:
        result = _to_str([1, 2, 3])
        assert result == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# _split_multi_url_content utility
# ---------------------------------------------------------------------------


class TestSplitMultiUrlContent:
    def test_single_section_with_prefix(self) -> None:
        content = "Content from: https://example.com\n# Hello World\nSome text."
        pairs = _split_multi_url_content(content, ["https://example.com"])
        assert len(pairs) == 1
        assert pairs[0][0] == "https://example.com"
        assert "Hello World" in pairs[0][1]

    def test_two_sections_split_correctly(self) -> None:
        content = (
            "Content from: https://a.com\nPage A content\n\n---\n\n"
            "Content from: https://b.com\nPage B content"
        )
        pairs = _split_multi_url_content(content, ["https://a.com", "https://b.com"])
        assert len(pairs) == 2
        assert pairs[0][0] == "https://a.com"
        assert "Page A" in pairs[0][1]
        assert pairs[1][0] == "https://b.com"
        assert "Page B" in pairs[1][1]

    def test_fallback_url_when_no_prefix(self) -> None:
        content = "Some content without a prefix"
        pairs = _split_multi_url_content(content, ["https://fallback.com"])
        assert len(pairs) == 1
        assert pairs[0][0] == "https://fallback.com"
        assert "Some content" in pairs[0][1]

    def test_empty_content_returns_empty_list(self) -> None:
        pairs = _split_multi_url_content("", [])
        assert pairs == []


# ---------------------------------------------------------------------------
# SpidraLoader — constructor validation
# ---------------------------------------------------------------------------


class TestSpidraLoaderInit:
    def test_raises_without_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Remove SPIDRA_API_KEY if present
            import os
            os.environ.pop("SPIDRA_API_KEY", None)
            with pytest.raises(ValueError, match="API key"):
                SpidraLoader("https://example.com")

    def test_raises_without_url_for_scrape_mode(self) -> None:
        with pytest.raises(ValueError, match="'url' is required"):
            SpidraLoader(api_key="spd_test")

    def test_raises_without_urls_for_batch_mode(self) -> None:
        with pytest.raises(ValueError, match="'urls' is required"):
            SpidraLoader(mode="batch", api_key="spd_test")

    def test_env_key_used_when_no_explicit_key(self) -> None:
        with patch.dict("os.environ", {"SPIDRA_API_KEY": "spd_env"}):
            with patch("langchain_spidra.document_loaders.Spidra"):
                loader = SpidraLoader("https://example.com")
                assert loader._url == "https://example.com"

    def test_url_wrapped_into_urls_list(self) -> None:
        with patch("langchain_spidra.document_loaders.Spidra"):
            loader = SpidraLoader("https://example.com", api_key="spd_test")
            assert loader._urls == ["https://example.com"]

    def test_explicit_urls_takes_precedence(self) -> None:
        with patch("langchain_spidra.document_loaders.Spidra"):
            loader = SpidraLoader(
                urls=["https://a.com", "https://b.com"],
                mode="batch",
                api_key="spd_test",
            )
            assert len(loader._urls) == 2


# ---------------------------------------------------------------------------
# SpidraLoader — scrape mode
# ---------------------------------------------------------------------------


class TestSpidraLoaderScrapeMode:
    def _make_loader(self, params: dict | None = None) -> SpidraLoader:
        with patch("langchain_spidra.document_loaders.Spidra"):
            return SpidraLoader("https://example.com", api_key="spd_test", params=params)

    def test_raw_markdown_yields_one_doc_per_url(self) -> None:
        loader = self._make_loader()
        content = (
            "Content from: https://example.com\n# Title\nBody text."
        )
        loader._client.scrape = MagicMock(return_value=_FakeScrapeResult(content=content))

        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "https://example.com"
        assert "Body text" in docs[0].page_content

    def test_raw_markdown_two_urls_yields_two_docs(self) -> None:
        with patch("langchain_spidra.document_loaders.Spidra"):
            loader = SpidraLoader(
                urls=["https://a.com", "https://b.com"],
                api_key="spd_test",
            )
        content = (
            "Content from: https://a.com\nA content\n\n---\n\n"
            "Content from: https://b.com\nB content"
        )
        loader._client.scrape = MagicMock(return_value=_FakeScrapeResult(content=content))

        docs = list(loader.lazy_load())
        assert len(docs) == 2
        assert docs[0].metadata["source"] == "https://a.com"
        assert docs[1].metadata["source"] == "https://b.com"

    def test_prompt_yields_single_doc_with_ai_result(self) -> None:
        loader = self._make_loader(params={"prompt": "Extract the title"})
        loader._client.scrape = MagicMock(
            return_value=_FakeScrapeResult(content={"title": "Hello"})
        )

        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert '"title"' in docs[0].page_content
        assert docs[0].metadata["ai_extraction_failed"] is False
        assert docs[0].metadata["extraction_empty"] is False

    def test_prompt_is_popped_from_params_not_passed_twice(self) -> None:
        loader = self._make_loader(params={"prompt": "Extract the title"})
        mock_scrape = MagicMock(return_value=_FakeScrapeResult(content="result"))
        loader._client.scrape = mock_scrape

        list(loader.lazy_load())

        call_args = mock_scrape.call_args
        scrape_params = call_args[0][0]  # first positional arg
        assert scrape_params.prompt == "Extract the title"

    def test_content_none_fallback_to_empty_string(self) -> None:
        loader = self._make_loader()
        loader._client.scrape = MagicMock(return_value=_FakeScrapeResult(content=None))

        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert docs[0].page_content == ""

    def test_at_most_3_urls_sent_to_scrape(self) -> None:
        with patch("langchain_spidra.document_loaders.Spidra"):
            loader = SpidraLoader(
                urls=["https://a.com", "https://b.com", "https://c.com", "https://d.com"],
                api_key="spd_test",
            )
        mock_scrape = MagicMock(return_value=_FakeScrapeResult(content=""))
        loader._client.scrape = mock_scrape

        list(loader.lazy_load())

        scrape_params = mock_scrape.call_args[0][0]
        assert len(scrape_params.urls) == 3


# ---------------------------------------------------------------------------
# SpidraLoader — crawl mode
# ---------------------------------------------------------------------------


class TestSpidraLoaderCrawlMode:
    def _make_loader(self, params: dict | None = None) -> SpidraLoader:
        with patch("langchain_spidra.document_loaders.Spidra"):
            return SpidraLoader(
                "https://docs.example.com",
                mode="crawl",
                api_key="spd_test",
                params=params,
            )

    def test_yields_one_doc_per_page(self) -> None:
        loader = self._make_loader()
        loader._client.crawl = MagicMock(
            return_value=_FakeCrawlJob(
                result=[
                    _FakeCrawlPage("https://docs.example.com/intro", "Intro page", "Introduction"),
                    _FakeCrawlPage("https://docs.example.com/api", "API reference", "API"),
                ]
            )
        )

        docs = list(loader.lazy_load())
        assert len(docs) == 2
        assert docs[0].metadata["source"] == "https://docs.example.com/intro"
        assert docs[0].metadata["title"] == "Introduction"
        assert docs[1].metadata["source"] == "https://docs.example.com/api"

    def test_page_with_no_title_has_no_title_metadata(self) -> None:
        loader = self._make_loader()
        loader._client.crawl = MagicMock(
            return_value=_FakeCrawlJob(
                result=[_FakeCrawlPage("https://example.com/page", "Content", None)]
            )
        )

        docs = list(loader.lazy_load())
        assert "title" not in docs[0].metadata

    def test_empty_data_page_skipped(self) -> None:
        loader = self._make_loader()
        loader._client.crawl = MagicMock(
            return_value=_FakeCrawlJob(
                result=[
                    _FakeCrawlPage("https://example.com/empty", None),
                    _FakeCrawlPage("https://example.com/real", "Real content"),
                ]
            )
        )

        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "https://example.com/real"

    def test_dict_data_json_serialised(self) -> None:
        loader = self._make_loader()
        loader._client.crawl = MagicMock(
            return_value=_FakeCrawlJob(
                result=[_FakeCrawlPage("https://example.com", {"key": "val"})]
            )
        )

        docs = list(loader.lazy_load())
        assert '"key"' in docs[0].page_content

    def test_custom_crawl_instruction_forwarded(self) -> None:
        loader = self._make_loader(
            params={"crawl_instruction": "Find blog posts", "transform_instruction": "Get body"}
        )
        mock_crawl = MagicMock(return_value=_FakeCrawlJob(result=[]))
        loader._client.crawl = mock_crawl

        list(loader.lazy_load())

        crawl_params = mock_crawl.call_args[0][0]
        assert crawl_params.crawl_instruction == "Find blog posts"
        assert crawl_params.transform_instruction == "Get body"

    def test_no_transform_instruction_defaults_to_none(self) -> None:
        loader = self._make_loader()
        mock_crawl = MagicMock(return_value=_FakeCrawlJob(result=[]))
        loader._client.crawl = mock_crawl

        list(loader.lazy_load())

        crawl_params = mock_crawl.call_args[0][0]
        assert crawl_params.transform_instruction is None


# ---------------------------------------------------------------------------
# SpidraLoader — batch mode
# ---------------------------------------------------------------------------


class TestSpidraLoaderBatchMode:
    def _make_loader(self, urls: list[str] | None = None) -> SpidraLoader:
        with patch("langchain_spidra.document_loaders.Spidra"):
            return SpidraLoader(
                urls=urls or ["https://a.com", "https://b.com"],
                mode="batch",
                api_key="spd_test",
            )

    def test_yields_one_doc_per_completed_item(self) -> None:
        loader = self._make_loader()
        loader._client.batch_scrape = MagicMock(
            return_value=_FakeBatchResponse(
                items=[
                    _FakeBatchItem("https://a.com", "Content A"),
                    _FakeBatchItem("https://b.com", "Content B"),
                ]
            )
        )

        docs = list(loader.lazy_load())
        assert len(docs) == 2
        assert docs[0].page_content == "Content A"
        assert docs[1].metadata["source"] == "https://b.com"

    def test_failed_items_skipped(self) -> None:
        loader = self._make_loader()
        loader._client.batch_scrape = MagicMock(
            return_value=_FakeBatchResponse(
                items=[
                    _FakeBatchItem("https://a.com", "Good content"),
                    _FakeBatchItem("https://b.com", None, status="failed"),
                ]
            )
        )

        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "https://a.com"

    def test_pending_items_skipped(self) -> None:
        loader = self._make_loader()
        loader._client.batch_scrape = MagicMock(
            return_value=_FakeBatchResponse(
                items=[_FakeBatchItem("https://a.com", None, status="pending")]
            )
        )

        docs = list(loader.lazy_load())
        assert len(docs) == 0

    def test_load_returns_list(self) -> None:
        loader = self._make_loader()
        loader._client.batch_scrape = MagicMock(
            return_value=_FakeBatchResponse(
                items=[_FakeBatchItem("https://a.com", "Content")]
            )
        )

        docs = loader.load()
        assert isinstance(docs, list)
        assert len(docs) == 1
