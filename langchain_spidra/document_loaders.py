"""Spidra document loader.

Supports scraping, crawling, and batch-scraping via the Spidra API. The
canonical class is :class:`SpidraLoader`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Literal

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_spidra._utilities import create_client, resolve_api_key


def _to_str(content: Any) -> str:
    """Normalise API content (str, dict, list, None) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


_SCRAPE_CONTENT_SEPARATOR = "\n\n---\n\n"
_CONTENT_FROM_PREFIX = "Content from: "


def _split_multi_url_content(content: str, urls: list[str]) -> list[tuple[str, str]]:
    """Split raw multi-URL scrape content into per-URL (source_url, text) pairs."""
    sections = content.split(_SCRAPE_CONTENT_SEPARATOR)
    result: list[tuple[str, str]] = []

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n", 1)
        first_line = lines[0].strip()

        if first_line.startswith(_CONTENT_FROM_PREFIX):
            source = first_line[len(_CONTENT_FROM_PREFIX) :].strip()
            text = lines[1].strip() if len(lines) > 1 else ""
        else:
            source = urls[i] if i < len(urls) else ""
            text = section

        result.append((source, text))

    return result


class SpidraLoader(BaseLoader):
    """Spidra document loader integration.

    Setup:
        Install ``langchain-spidra`` and set the environment variable
        ``SPIDRA_API_KEY``.

        ```bash
        pip install -U langchain-spidra
        export SPIDRA_API_KEY="your-api-key"
        ```

    Instantiate:
        ```python
        from langchain_spidra import SpidraLoader

        # Scrape a single page
        loader = SpidraLoader(url="https://example.com", api_key="spd_...")

        # Crawl an entire site
        loader = SpidraLoader(url="https://docs.example.com", mode="crawl")

        # Batch scrape multiple URLs
        loader = SpidraLoader(urls=["https://a.com", "https://b.com"], mode="batch")
        ```

    Lazy load:
        ```python
        for doc in loader.lazy_load():
            print(doc.page_content[:100])
            print(doc.metadata)
        ```
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        urls: list[str] | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        mode: Literal["scrape", "crawl", "batch"] = "scrape",
        params: dict | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            url: The URL to scrape or crawl (required for ``scrape``/``crawl``
                modes).
            urls: List of URLs (for ``batch`` mode or multi-URL scrape).
            api_key: Spidra API key. Falls back to ``SPIDRA_API_KEY`` env var.
            api_url: Override the base API URL.
            mode: One of ``"scrape"`` (1–3 URLs), ``"crawl"`` (whole site), or
                ``"batch"`` (up to 50 URLs in parallel).
            params: Extra parameters forwarded to the underlying SDK method.
        """
        resolved_key = resolve_api_key(api_key)
        if not resolved_key:
            msg = "Spidra API key not found. Pass api_key= or set SPIDRA_API_KEY."
            raise ValueError(msg)

        if mode in ("scrape", "crawl") and not url and not urls:
            msg = f"'url' is required for mode='{mode}'"
            raise ValueError(msg)
        if mode == "batch" and not urls and not url:
            msg = "'urls' is required for mode='batch'"
            raise ValueError(msg)

        self._url = url
        self._urls: list[str] = urls or ([url] if url else [])
        self._mode = mode
        self._params: dict[str, Any] = params or {}
        self._client = create_client(api_key=resolved_key, api_url=api_url)

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from Spidra, one per result."""
        if self._mode == "scrape":
            yield from self._load_scrape()
        elif self._mode == "crawl":
            yield from self._load_crawl()
        elif self._mode == "batch":
            yield from self._load_batch()
        else:
            msg = f"Unknown mode: {self._mode!r}"
            raise ValueError(msg)

    def _load_scrape(self) -> Iterator[Document]:
        from spidra import ScrapeParams, ScrapeUrl

        params = dict(self._params)
        prompt = params.pop("prompt", None)

        scrape_urls = [ScrapeUrl(url=u) for u in self._urls[:3]]
        result = self._client.scrape(ScrapeParams(urls=scrape_urls, prompt=prompt, **params))

        content_str = _to_str(result.content)

        if prompt:
            source = self._urls[0] if len(self._urls) == 1 else ", ".join(self._urls[:3])
            yield Document(
                page_content=content_str,
                metadata={
                    "source": source,
                    "ai_extraction_failed": result.ai_extraction_failed,
                    "extraction_empty": result.extraction_empty,
                },
            )
            return

        pairs = _split_multi_url_content(content_str, self._urls)
        if pairs:
            for source, text in pairs:
                yield Document(page_content=text, metadata={"source": source})
        else:
            yield Document(
                page_content=content_str,
                metadata={"source": self._urls[0] if self._urls else ""},
            )

    def _load_crawl(self) -> Iterator[Document]:
        from spidra import CrawlParams

        params = dict(self._params)
        crawl_instruction = params.pop("crawl_instruction", "Find all pages")
        transform_instruction = params.pop("transform_instruction", None)

        base_url = self._url or self._urls[0]
        job = self._client.crawl(
            CrawlParams(
                base_url=base_url,
                crawl_instruction=crawl_instruction,
                transform_instruction=transform_instruction,
                **params,
            )
        )

        for page in job.result:
            text = _to_str(page.data)
            if not text:
                continue
            metadata: dict[str, Any] = {"source": page.url}
            if page.title:
                metadata["title"] = page.title
            yield Document(page_content=text, metadata=metadata)

    def _load_batch(self) -> Iterator[Document]:
        from spidra import BatchScrapeParams

        params = dict(self._params)
        batch = self._client.batch_scrape(BatchScrapeParams(urls=self._urls, **params))

        for item in batch.items:
            if item.status != "completed" or item.result is None:
                continue
            yield Document(
                page_content=_to_str(item.result),
                metadata={"source": item.url},
            )
