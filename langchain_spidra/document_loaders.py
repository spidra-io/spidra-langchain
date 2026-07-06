from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Literal, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from spidra import (
    Spidra,
    ScrapeParams,
    ScrapeUrl,
    CrawlParams,
    BatchScrapeParams,
    PollOptions,
)

_SCRAPE_CONTENT_SEPARATOR = "\n\n---\n\n"
_CONTENT_FROM_PREFIX = "Content from: "


def _to_str(content: Any) -> str:
    """Normalise API content (str, dict, list, None) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _poll_kwargs(po: Optional[PollOptions]) -> Dict[str, Any]:
    """Convert a PollOptions object to keyword arguments for the new client API."""
    if po is None:
        return {}
    return {"poll_interval": po.poll_interval, "timeout": po.timeout}


def _split_multi_url_content(content: str, urls: List[str]) -> List[tuple[str, str]]:
    """
    Split raw multi-URL scrape content into per-URL (source_url, text) pairs.

    The worker concatenates sections using the format:
        Content from: https://example.com
        [markdown body]

        ---

        Content from: https://other.com
        [markdown body]
    """
    sections = content.split(_SCRAPE_CONTENT_SEPARATOR)
    result: List[tuple[str, str]] = []

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n", 1)
        first_line = lines[0].strip()

        if first_line.startswith(_CONTENT_FROM_PREFIX):
            source = first_line[len(_CONTENT_FROM_PREFIX):].strip()
            text = lines[1].strip() if len(lines) > 1 else ""
        else:
            source = urls[i] if i < len(urls) else ""
            text = section

        result.append((source, text))

    return result


class SpidraLoader(BaseLoader):
    """
    LangChain document loader backed by the Spidra web scraping API.

    Supports three modes:

    - ``scrape``: Scrape 1–3 URLs. Without a ``prompt``, returns raw markdown
      as one Document per URL (ideal for RAG indexing). With a ``prompt``,
      returns a single Document containing the AI-extracted result.

    - ``crawl``: Crawl an entire site from ``url``, yielding one Document per
      discovered page.

    - ``batch``: Scrape 1–50 URLs in parallel, yielding one Document per URL.

    Args:
        url: The URL to scrape or crawl (required for ``scrape`` / ``crawl`` modes).
        urls: List of URLs (used for ``batch`` mode or multi-URL scrape).
        api_key: Spidra API key. Falls back to ``SPIDRA_API_KEY`` env var.
        api_url: Override the base API URL (defaults to ``https://api.spidra.io/api``).
        mode: One of ``"scrape"``, ``"crawl"``, or ``"batch"``.
        params: Extra parameters forwarded to the underlying SDK method.
            For scrape/batch: ``prompt``, ``output``, ``use_proxy``, ``proxy_country``,
            ``extract_content_only``, ``schema``, ``screenshot``, etc.
            For crawl: ``crawl_instruction``, ``transform_instruction``, ``max_pages``,
            ``use_proxy``, ``proxy_country``, etc.
        poll_options: Override polling behaviour (timeout, interval).

    Usage::

        # Raw markdown — best for RAG / vector store indexing
        loader = SpidraLoader("https://example.com", api_key="spd_...")
        docs = loader.load()

        # AI extraction
        loader = SpidraLoader(
            "https://example.com",
            api_key="spd_...",
            params={"prompt": "Extract product names and prices", "output": "json"},
        )

        # Crawl a whole docs site
        loader = SpidraLoader(
            "https://docs.example.com",
            mode="crawl",
            api_key="spd_...",
            params={
                "crawl_instruction": "Find all documentation pages",
                "transform_instruction": "Return the page content as clean markdown",
                "max_pages": 50,
            },
        )

        # Batch scrape many URLs
        loader = SpidraLoader(
            urls=["https://a.com", "https://b.com", "https://c.com"],
            mode="batch",
            api_key="spd_...",
        )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        urls: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: Literal["scrape", "crawl", "batch"] = "scrape",
        params: Optional[Dict[str, Any]] = None,
        poll_options: Optional[PollOptions] = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("SPIDRA_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Spidra API key not found. Pass api_key= or set SPIDRA_API_KEY."
            )

        if mode in ("scrape", "crawl") and not url and not urls:
            raise ValueError(f"'url' is required for mode='{mode}'")
        if mode == "batch" and not urls and not url:
            raise ValueError("'urls' is required for mode='batch'")

        self._url = url
        self._urls: List[str] = urls or ([url] if url else [])
        self._mode = mode
        self._params: Dict[str, Any] = params or {}
        self._poll_options = poll_options
        self._client = Spidra(api_key=resolved_key, api_url=api_url)

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        if self._mode == "scrape":
            yield from self._load_scrape()
        elif self._mode == "crawl":
            yield from self._load_crawl()
        elif self._mode == "batch":
            yield from self._load_batch()
        else:
            raise ValueError(f"Unknown mode: {self._mode!r}")

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _load_scrape(self) -> Iterator[Document]:
        params = dict(self._params)
        prompt = params.pop("prompt", None)

        scrape_urls = [ScrapeUrl(url=u) for u in self._urls[:3]]
        result = self._client.scrape(
            ScrapeParams(urls=scrape_urls, prompt=prompt, **params),
            **_poll_kwargs(self._poll_options),
        )

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
            ),
            **_poll_kwargs(self._poll_options),
        )

        for page in job.result:
            text = _to_str(page.data)
            if not text:
                continue
            metadata: Dict[str, Any] = {"source": page.url}
            if page.title:
                metadata["title"] = page.title
            yield Document(page_content=text, metadata=metadata)

    def _load_batch(self) -> Iterator[Document]:
        params = dict(self._params)

        batch = self._client.batch_scrape(
            BatchScrapeParams(urls=self._urls, **params),
            **_poll_kwargs(self._poll_options),
        )

        for item in batch.items:
            if item.status != "completed" or item.result is None:
                continue
            yield Document(
                page_content=_to_str(item.result),
                metadata={"source": item.url},
            )
