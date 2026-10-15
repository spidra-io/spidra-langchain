"""Spidra document loader.

Supports scraping, crawling, batch-scraping, search, mapping, and document parsing
via the Spidra API. The canonical class is :class:`SpidraLoader`.
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

        # Parse a PDF or document into markdown
        loader = SpidraLoader(source="https://example.com/document.pdf", mode="parse")

        # Search the web and load results
        loader = SpidraLoader(query="artificial intelligence", mode="search")

        # Discover all URLs on a site
        loader = SpidraLoader(url="https://docs.example.com", mode="map")
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
        query: str | None = None,
        source: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        mode: Literal["scrape", "crawl", "batch", "search", "map", "parse"] = "scrape",
        params: dict | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            url: The URL to scrape, crawl, or map.
            urls: List of URLs (for ``batch`` mode or multi-URL scrape).
            query: The search query (for ``search`` mode).
            source: Document source file path or URL (for ``parse`` mode).
            api_key: Spidra API key. Falls back to ``SPIDRA_API_KEY`` env var.
            api_url: Override the base API URL.
            mode: One of ``"scrape"``, ``"crawl"``, ``"batch"``, ``"search"``,
                ``"map"``, or ``"parse"``.
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
        if mode == "search" and not query and not url:
            msg = "'query' is required for mode='search'"
            raise ValueError(msg)
        if mode == "map" and not url and not urls:
            msg = "'url' is required for mode='map'"
            raise ValueError(msg)
        if mode == "parse" and not source and not url and not urls:
            msg = "'source' is required for mode='parse'"
            raise ValueError(msg)

        self._url = url
        self._urls: list[str] = urls or ([url] if url else [])
        self._query = query or (url if mode == "search" else None)
        self._source = source or url or (self._urls[0] if self._urls else None)
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
        elif self._mode == "search":
            yield from self._load_search()
        elif self._mode == "map":
            yield from self._load_map()
        elif self._mode == "parse":
            yield from self._load_parse()
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
            text = _to_str(getattr(page, "data", None))
            if not text:
                continue
            metadata: dict[str, Any] = {"source": page.url}
            if getattr(page, "title", None):
                metadata["title"] = page.title
            if getattr(page, "depth", None) is not None:
                metadata["depth"] = page.depth
            if getattr(page, "parent_url", None):
                metadata["parent_url"] = page.parent_url
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

    def _load_search(self) -> Iterator[Document]:
        params = dict(self._params)
        q = self._query or self._url or ""
        result = self._client.search(q, **params)
        data = getattr(result, "data", result)
        for src in ("web", "news", "research", "developer", "images", "videos"):
            items = getattr(data, src, None)
            if items:
                for item in items:
                    content = (
                        getattr(item, "markdown", None)
                        or getattr(item, "description", None)
                        or getattr(item, "title", "")
                        or ""
                    )
                    metadata: dict[str, Any] = {
                        "source": getattr(item, "url", ""),
                        "title": getattr(item, "title", ""),
                        "source_type": src,
                    }
                    if getattr(item, "date", None):
                        metadata["date"] = item.date
                    if getattr(item, "authors", None):
                        metadata["authors"] = item.authors
                    if getattr(item, "doi", None):
                        metadata["doi"] = item.doi
                    yield Document(page_content=str(content), metadata=metadata)

    def _load_map(self) -> Iterator[Document]:
        params = dict(self._params)
        target_url = self._url or (self._urls[0] if self._urls else "")
        result = self._client.map(target_url, **params)
        for link in getattr(result, "links", []):
            metadata: dict[str, Any] = {"source": link.url}
            if getattr(link, "title", None):
                metadata["title"] = link.title
            yield Document(page_content=link.url, metadata=metadata)

    def _load_parse(self) -> Iterator[Document]:
        params = dict(self._params)
        params.setdefault("include_pages", True)
        src = self._source or self._url or (self._urls[0] if self._urls else "")
        result = self._client.parse(src, **params)
        pages = getattr(result, "pages", None)
        if pages:
            for p in pages:
                metadata = {
                    "source": getattr(result, "source_url", None)
                    or getattr(result, "file_name", None)
                    or src,
                    "page_number": p.page_number,
                    "method": getattr(p, "method", "text"),
                }
                if getattr(result, "total_pages", None):
                    metadata["total_pages"] = result.total_pages
                yield Document(page_content=p.markdown, metadata=metadata)
        elif getattr(result, "markdown", None):
            metadata = {
                "source": getattr(result, "source_url", None)
                or getattr(result, "file_name", None)
                or src,
            }
            if getattr(result, "total_pages", None):
                metadata["total_pages"] = result.total_pages
            yield Document(page_content=str(result.markdown or ""), metadata=metadata)
        elif getattr(result, "json", None) is not None:
            metadata = {
                "source": getattr(result, "source_url", None)
                or getattr(result, "file_name", None)
                or src,
            }
            yield Document(
                page_content=json.dumps(result.json, ensure_ascii=False), metadata=metadata
            )


class SpidraSearchLoader(SpidraLoader):
    """Convenience loader for Spidra Search results."""

    def __init__(
        self,
        query: str,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        params: dict | None = None,
        **extra_params: Any,
    ) -> None:
        merged_params = dict(params or {})
        merged_params.update(extra_params)
        super().__init__(
            query=query,
            api_key=api_key,
            api_url=api_url,
            mode="search",
            params=merged_params,
        )


class SpidraParseLoader(SpidraLoader):
    """Convenience loader for parsing documents (PDF, DOCX, etc.) via Spidra Parse."""

    def __init__(
        self,
        source: str,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        params: dict | None = None,
        **extra_params: Any,
    ) -> None:
        merged_params = dict(params or {})
        merged_params.update(extra_params)
        super().__init__(
            source=source,
            api_key=api_key,
            api_url=api_url,
            mode="parse",
            params=merged_params,
        )


class SpidraMapLoader(SpidraLoader):
    """Convenience loader for website structure via Spidra Map."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        params: dict | None = None,
        **extra_params: Any,
    ) -> None:
        merged_params = dict(params or {})
        merged_params.update(extra_params)
        super().__init__(
            url=url,
            api_key=api_key,
            api_url=api_url,
            mode="map",
            params=merged_params,
        )
