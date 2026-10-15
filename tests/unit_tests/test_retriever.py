"""Unit tests for ``SpidraRetriever`` with mocked client."""

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_spidra.retrievers import SpidraRetriever


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


def test_retriever_get_relevant_documents() -> None:
    """SpidraRetriever returns Documents from search results."""
    retriever = SpidraRetriever(api_key="spd-test", limit=5)  # type: ignore[arg-type]
    retriever.client = MagicMock()
    retriever.client.search.return_value = _FakeSearchResult(
        data=_FakeSearchData(
            web=[
                _FakeSearchResultItem(
                    title="Doc 1",
                    url="https://example.com/1",
                    markdown="# Markdown 1",
                ),
                _FakeSearchResultItem(
                    title="Doc 2",
                    url="https://example.com/2",
                    description="Snippet 2",
                ),
            ]
        )
    )

    docs = retriever.invoke("machine learning")

    assert len(docs) == 2
    assert docs[0].page_content == "# Markdown 1"
    assert docs[0].metadata["source"] == "https://example.com/1"
    assert docs[0].metadata["title"] == "Doc 1"

    assert docs[1].page_content == "Snippet 2"
    assert docs[1].metadata["source"] == "https://example.com/2"


@pytest.mark.asyncio
async def test_retriever_aget_relevant_documents() -> None:
    """Async retrieval with SpidraRetriever."""
    retriever = SpidraRetriever(api_key="spd-test", limit=3)  # type: ignore[arg-type]
    retriever.async_client = MagicMock()
    retriever.async_client.search = AsyncMock(
        return_value=_FakeSearchResult(
            data=_FakeSearchData(
                web=[
                    _FakeSearchResultItem(
                        title="Async Doc",
                        url="https://example.com/async",
                        description="Async snippet",
                    )
                ]
            )
        )
    )

    docs = await retriever.ainvoke("async test")
    assert len(docs) == 1
    assert docs[0].page_content == "Async snippet"
    assert docs[0].metadata["source"] == "https://example.com/async"
