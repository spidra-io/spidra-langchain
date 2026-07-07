"""Tools for the Spidra API.

Each capability of the Spidra API is exposed as a ``BaseTool`` so it can be
bound to an agent: scrape, crawl, and batch-scrape. Responses are normalized
into JSON-friendly dicts from the typed dataclasses returned by the
``spidra`` SDK.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator
from spidra import Spidra

from langchain_spidra._utilities import clean_params, initialize_client


class _SpidraBaseTool(BaseTool):
    """Shared configuration for Spidra tools (client + credentials)."""

    client: Spidra = Field(default=None)  # type: ignore[assignment]
    api_key: SecretStr = Field(default=SecretStr(""))
    api_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Resolve credentials and instantiate the Spidra client."""
        return initialize_client(values)


class SpidraScrape(_SpidraBaseTool):
    """Scrape one or more URLs and return their content as markdown.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

        ```bash
        pip install -U langchain-spidra
        export SPIDRA_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_spidra import SpidraScrape

        tool = SpidraScrape()
        tool.invoke({"url": "https://example.com"})
        ```
    """

    name: str = "spidra_scrape"
    description: str = (
        "Scrape a web page and return its content as clean markdown. "
        "Optionally provide a 'prompt' to run AI extraction and get structured "
        "data. Input should be a URL."
    )

    def _run(
        self,
        url: str,
        prompt: str | None = None,
        output: str | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        extract_content_only: bool | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Scrape ``url``.

        Args:
            url: The URL to scrape.
            prompt: Natural-language extraction instruction (enables AI mode).
            output: Desired output format (e.g. ``"json"``).
            use_proxy: Whether to route the request through a proxy.
            proxy_country: ISO 3166-1 alpha-2 country code for the proxy.
            extract_content_only: Strip navigation/ads and return main content only.
            run_manager: The run manager for callbacks.

        Returns:
            A dict with ``content``, ``ai_extraction_failed``, and
            ``extraction_empty``, or an error string.
        """
        try:
            result = self.client.scrape(
                url,
                **clean_params(
                    {
                        "prompt": prompt,
                        "output": output,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                        "extract_content_only": extract_content_only,
                    }
                ),
            )
            content = result.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False) if content else ""
            return {
                "content": content,
                "ai_extraction_failed": result.ai_extraction_failed,
                "extraction_empty": result.extraction_empty,
            }
        except Exception as e:
            return repr(e)


class SpidraCrawl(_SpidraBaseTool):
    """Crawl a site starting from a URL and return each page as markdown.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraCrawl

        tool = SpidraCrawl()
        tool.invoke({"url": "https://docs.example.com", "max_pages": 20})
        ```
    """

    name: str = "spidra_crawl"
    description: str = (
        "Crawl a website starting from a URL, following accessible sub-pages, "
        "and return each page's content as markdown. "
        "Use 'max_pages' to cap the number of pages crawled."
    )

    def _run(
        self,
        url: str,
        crawl_instruction: str | None = None,
        transform_instruction: str | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]] | str:
        """Crawl from ``url``.

        Args:
            url: The URL to start crawling from.
            crawl_instruction: Natural-language instruction for which pages to follow.
            transform_instruction: What to extract from each page (enables AI mode).
            max_pages: Maximum number of pages to crawl.
            max_depth: Maximum link depth from the starting URL.
            include_paths: URL path patterns to include.
            exclude_paths: URL path patterns to skip.
            use_proxy: Whether to route requests through a proxy.
            proxy_country: ISO 3166-1 alpha-2 country code for the proxy.
            run_manager: The run manager for callbacks.

        Returns:
            A list of page dicts (``url``, ``data``, ``title``), or an error string.
        """
        try:
            job = self.client.crawl(
                url,
                **clean_params(
                    {
                        "crawl_instruction": crawl_instruction,
                        "transform_instruction": transform_instruction,
                        "max_pages": max_pages,
                        "max_depth": max_depth,
                        "include_paths": include_paths,
                        "exclude_paths": exclude_paths,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                    }
                ),
            )
            return [
                {
                    "url": page.url,
                    "data": page.data or "",
                    "title": page.title,
                }
                for page in job.result
            ]
        except Exception as e:
            return repr(e)


class SpidraBatchScrape(_SpidraBaseTool):
    """Scrape a list of URLs in parallel and return each page's content.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraBatchScrape

        tool = SpidraBatchScrape()
        tool.invoke({"urls": ["https://a.com", "https://b.com"]})
        ```
    """

    name: str = "spidra_batch_scrape"
    description: str = (
        "Scrape a list of URLs in parallel and return each page's content. "
        "Input should be a list of URLs. Optionally provide a 'prompt' for "
        "AI extraction across all pages."
    )

    def _run(
        self,
        urls: list[str],
        prompt: str | None = None,
        output: str | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        extract_content_only: bool | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]] | str:
        """Batch-scrape ``urls``.

        Args:
            urls: The list of URLs to scrape.
            prompt: Natural-language extraction instruction (enables AI mode).
            output: Desired output format (e.g. ``"json"``).
            use_proxy: Whether to route requests through a proxy.
            proxy_country: ISO 3166-1 alpha-2 country code for the proxy.
            extract_content_only: Strip navigation/ads and return main content only.
            run_manager: The run manager for callbacks.

        Returns:
            A list of item dicts (``url``, ``result``, ``status``), or an error string.
        """
        try:
            batch = self.client.batch_scrape(
                urls,
                **clean_params(
                    {
                        "prompt": prompt,
                        "output": output,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                        "extract_content_only": extract_content_only,
                    }
                ),
            )
            return [
                {
                    "url": item.url,
                    "result": item.result or "",
                    "status": item.status,
                }
                for item in batch.items
            ]
        except Exception as e:
            return repr(e)
