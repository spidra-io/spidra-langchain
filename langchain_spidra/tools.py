"""Tools for the Spidra API.

Each capability of the Spidra API is exposed as a ``BaseTool`` so it can be
bound to an agent: scrape, crawl, batch-scrape, search, map, parse, and bulk-scrape.
Responses are normalized into JSON-friendly dicts from the typed dataclasses
returned by the ``spidra`` SDK.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator
from spidra import AsyncSpidra, Spidra

from langchain_spidra._utilities import clean_params, initialize_client, to_dict


class _SpidraBaseTool(BaseTool):
    """Shared configuration for Spidra tools (client + credentials)."""

    client: Spidra = Field(default=None)  # type: ignore[assignment]
    async_client: AsyncSpidra = Field(default=None)  # type: ignore[assignment]
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
        schema: dict[str, Any] | None = None,
        output: str | None = None,
        scrape_mode: str | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        extract_content_only: bool | None = None,
        screenshot: bool | None = None,
        full_page_screenshot: bool | None = None,
        cookies: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Scrape ``url`` synchronously."""
        try:
            result = self.client.scrape(
                url,
                **clean_params(
                    {
                        "prompt": prompt,
                        "schema": schema,
                        "output": output,
                        "scrape_mode": scrape_mode,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                        "extract_content_only": extract_content_only,
                        "screenshot": screenshot,
                        "full_page_screenshot": full_page_screenshot,
                        "cookies": cookies,
                    }
                ),
            )
            content = result.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False) if content else ""
            res: dict[str, Any] = {
                "content": content,
                "ai_extraction_failed": result.ai_extraction_failed,
                "extraction_empty": result.extraction_empty,
            }
            if getattr(result, "screenshots", None):
                res["screenshots"] = result.screenshots
            return res
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        url: str,
        prompt: str | None = None,
        schema: dict[str, Any] | None = None,
        output: str | None = None,
        scrape_mode: str | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        extract_content_only: bool | None = None,
        screenshot: bool | None = None,
        full_page_screenshot: bool | None = None,
        cookies: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Scrape ``url`` asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(
                    self._run,
                    url=url,
                    prompt=prompt,
                    schema=schema,
                    output=output,
                    scrape_mode=scrape_mode,
                    use_proxy=use_proxy,
                    proxy_country=proxy_country,
                    extract_content_only=extract_content_only,
                    screenshot=screenshot,
                    full_page_screenshot=full_page_screenshot,
                    cookies=cookies,
                )
            result = await client.scrape(
                url,
                **clean_params(
                    {
                        "prompt": prompt,
                        "schema": schema,
                        "output": output,
                        "scrape_mode": scrape_mode,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                        "extract_content_only": extract_content_only,
                        "screenshot": screenshot,
                        "full_page_screenshot": full_page_screenshot,
                        "cookies": cookies,
                    }
                ),
            )
            content = result.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False) if content else ""
            res_async: dict[str, Any] = {
                "content": content,
                "ai_extraction_failed": result.ai_extraction_failed,
                "extraction_empty": result.extraction_empty,
            }
            if getattr(result, "screenshots", None):
                res_async["screenshots"] = result.screenshots
            return res_async
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
        schema: dict[str, Any] | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        allow_subdomains: bool | None = None,
        crawl_entire_domain: bool | None = None,
        ignore_query_params: bool | None = None,
        respect_robots: bool | None = None,
        use_sitemap: bool | None = None,
        scrape_mode: str | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        cookies: str | None = None,
        webhook_url: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]] | str:
        """Crawl from ``url`` synchronously."""
        try:
            job = self.client.crawl(
                url,
                **clean_params(
                    {
                        "crawl_instruction": crawl_instruction,
                        "transform_instruction": transform_instruction,
                        "schema": schema,
                        "max_pages": max_pages,
                        "max_depth": max_depth,
                        "include_paths": include_paths,
                        "exclude_paths": exclude_paths,
                        "allow_subdomains": allow_subdomains,
                        "crawl_entire_domain": crawl_entire_domain,
                        "ignore_query_params": ignore_query_params,
                        "respect_robots": respect_robots,
                        "use_sitemap": use_sitemap,
                        "scrape_mode": scrape_mode,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                        "cookies": cookies,
                        "webhook_url": webhook_url,
                    }
                ),
            )
            return [
                {
                    "url": page.url,
                    "data": page.data or "",
                    "title": page.title,
                    "depth": getattr(page, "depth", None),
                    "parent_url": getattr(page, "parent_url", None),
                }
                for page in job.result
            ]
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        url: str,
        crawl_instruction: str | None = None,
        transform_instruction: str | None = None,
        schema: dict[str, Any] | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        allow_subdomains: bool | None = None,
        crawl_entire_domain: bool | None = None,
        ignore_query_params: bool | None = None,
        respect_robots: bool | None = None,
        use_sitemap: bool | None = None,
        scrape_mode: str | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        cookies: str | None = None,
        webhook_url: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]] | str:
        """Crawl from ``url`` asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(
                    self._run,
                    url=url,
                    crawl_instruction=crawl_instruction,
                    transform_instruction=transform_instruction,
                    schema=schema,
                    max_pages=max_pages,
                    max_depth=max_depth,
                    include_paths=include_paths,
                    exclude_paths=exclude_paths,
                    allow_subdomains=allow_subdomains,
                    crawl_entire_domain=crawl_entire_domain,
                    ignore_query_params=ignore_query_params,
                    respect_robots=respect_robots,
                    use_sitemap=use_sitemap,
                    scrape_mode=scrape_mode,
                    use_proxy=use_proxy,
                    proxy_country=proxy_country,
                    cookies=cookies,
                    webhook_url=webhook_url,
                )
            job = await client.crawl(
                url,
                **clean_params(
                    {
                        "crawl_instruction": crawl_instruction,
                        "transform_instruction": transform_instruction,
                        "schema": schema,
                        "max_pages": max_pages,
                        "max_depth": max_depth,
                        "include_paths": include_paths,
                        "exclude_paths": exclude_paths,
                        "allow_subdomains": allow_subdomains,
                        "crawl_entire_domain": crawl_entire_domain,
                        "ignore_query_params": ignore_query_params,
                        "respect_robots": respect_robots,
                        "use_sitemap": use_sitemap,
                        "scrape_mode": scrape_mode,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                        "cookies": cookies,
                        "webhook_url": webhook_url,
                    }
                ),
            )
            return [
                {
                    "url": page.url,
                    "data": page.data or "",
                    "title": page.title,
                    "depth": getattr(page, "depth", None),
                    "parent_url": getattr(page, "parent_url", None),
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
        schema: dict[str, Any] | None = None,
        output: str | None = None,
        scrape_mode: str | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        extract_content_only: bool | None = None,
        screenshot: bool | None = None,
        full_page_screenshot: bool | None = None,
        cookies: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]] | str:
        """Batch-scrape ``urls`` synchronously."""
        try:
            batch = self.client.batch_scrape(
                urls,
                **clean_params(
                    {
                        "prompt": prompt,
                        "schema": schema,
                        "output": output,
                        "scrape_mode": scrape_mode,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                        "extract_content_only": extract_content_only,
                        "screenshot": screenshot,
                        "full_page_screenshot": full_page_screenshot,
                        "cookies": cookies,
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

    async def _arun(
        self,
        urls: list[str],
        prompt: str | None = None,
        schema: dict[str, Any] | None = None,
        output: str | None = None,
        scrape_mode: str | None = None,
        use_proxy: bool | None = None,
        proxy_country: str | None = None,
        extract_content_only: bool | None = None,
        screenshot: bool | None = None,
        full_page_screenshot: bool | None = None,
        cookies: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]] | str:
        """Batch-scrape ``urls`` asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(
                    self._run,
                    urls=urls,
                    prompt=prompt,
                    schema=schema,
                    output=output,
                    scrape_mode=scrape_mode,
                    use_proxy=use_proxy,
                    proxy_country=proxy_country,
                    extract_content_only=extract_content_only,
                    screenshot=screenshot,
                    full_page_screenshot=full_page_screenshot,
                    cookies=cookies,
                )
            batch = await client.batch_scrape(
                urls,
                **clean_params(
                    {
                        "prompt": prompt,
                        "schema": schema,
                        "output": output,
                        "scrape_mode": scrape_mode,
                        "use_proxy": use_proxy,
                        "proxy_country": proxy_country,
                        "extract_content_only": extract_content_only,
                        "screenshot": screenshot,
                        "full_page_screenshot": full_page_screenshot,
                        "cookies": cookies,
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


class SpidraSearch(_SpidraBaseTool):
    """Search the web, news, academic research, or developer repositories.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraSearch

        tool = SpidraSearch()
        tool.invoke({"query": "latest artificial intelligence breakthroughs", "limit": 5})
        ```
    """

    name: str = "spidra_search"
    description: str = (
        "Search the web, news, academic research, or developer repositories using Spidra Search. "
        "Returns ranked results with titles, URLs, descriptions, and optional markdown content. "
        "Input should include a search 'query'."
    )

    def _format_search_response(self, query: str, result: Any) -> dict[str, Any]:
        results_list: list[dict[str, Any]] = []
        data = getattr(result, "data", result)
        for src in ("web", "news", "research", "developer", "images", "videos"):
            items = getattr(data, src, None)
            if items:
                for item in items:
                    results_list.append(
                        {
                            "title": getattr(item, "title", ""),
                            "url": getattr(item, "url", ""),
                            "content": getattr(item, "markdown", None)
                            or getattr(item, "description", None)
                            or "",
                            "source_type": src,
                            "source": getattr(item, "source", None),
                            "date": getattr(item, "date", None),
                            "authors": getattr(item, "authors", None),
                            "doi": getattr(item, "doi", None),
                        }
                    )
        extras = getattr(data, "extras", None)
        return {
            "query": query,
            "results": results_list,
            "extras": to_dict(extras) if extras else {},
        }

    def _run(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int | None = None,
        page: int | None = None,
        country: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        time_range: str | None = None,
        filetype: str | None = None,
        scrape_options: dict[str, Any] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Run search synchronously."""
        try:
            params = clean_params(
                {
                    "sources": sources,
                    "limit": limit,
                    "page": page,
                    "country": country,
                    "include_domains": include_domains,
                    "exclude_domains": exclude_domains,
                    "time_range": time_range,
                    "filetype": filetype,
                    "scrape_options": scrape_options,
                }
            )
            result = self.client.search(query, **params)
            return self._format_search_response(query, result)
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int | None = None,
        page: int | None = None,
        country: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        time_range: str | None = None,
        filetype: str | None = None,
        scrape_options: dict[str, Any] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Run search asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(
                    self._run,
                    query=query,
                    sources=sources,
                    limit=limit,
                    page=page,
                    country=country,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    time_range=time_range,
                    filetype=filetype,
                    scrape_options=scrape_options,
                )
            params = clean_params(
                {
                    "sources": sources,
                    "limit": limit,
                    "page": page,
                    "country": country,
                    "include_domains": include_domains,
                    "exclude_domains": exclude_domains,
                    "time_range": time_range,
                    "filetype": filetype,
                    "scrape_options": scrape_options,
                }
            )
            result = await client.search(query, **params)
            return self._format_search_response(query, result)
        except Exception as e:
            return repr(e)


class SpidraMap(_SpidraBaseTool):
    """Map a website to discover all its sub-pages and URLs without scraping.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraMap

        tool = SpidraMap()
        tool.invoke({"url": "https://docs.example.com", "limit": 100})
        ```
    """

    name: str = "spidra_map"
    description: str = (
        "Map a website to discover all its sub-pages and URLs using sitemaps and page links "
        "without scraping. Useful for exploring site structure and discovering links to crawl. "
        "Input should include a 'url'."
    )

    def _run(
        self,
        url: str,
        search: str | None = None,
        limit: int | None = None,
        sitemap: str | None = None,
        include_subdomains: bool | None = None,
        ignore_query_params: bool | None = None,
        respect_robots: bool | None = None,
        cookies: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Map website synchronously."""
        try:
            params = clean_params(
                {
                    "search": search,
                    "limit": limit,
                    "sitemap": sitemap,
                    "include_subdomains": include_subdomains,
                    "ignore_query_params": ignore_query_params,
                    "respect_robots": respect_robots,
                    "cookies": cookies,
                }
            )
            result = self.client.map(url, **params)
            return {
                "links": [
                    {"url": link.url, "title": getattr(link, "title", None)}
                    for link in getattr(result, "links", [])
                ],
                "total": getattr(result, "total", 0),
                "sources": getattr(result, "sources", {}),
                "credits_used": getattr(result, "credits_used", 0),
            }
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        url: str,
        search: str | None = None,
        limit: int | None = None,
        sitemap: str | None = None,
        include_subdomains: bool | None = None,
        ignore_query_params: bool | None = None,
        respect_robots: bool | None = None,
        cookies: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Map website asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(
                    self._run,
                    url=url,
                    search=search,
                    limit=limit,
                    sitemap=sitemap,
                    include_subdomains=include_subdomains,
                    ignore_query_params=ignore_query_params,
                    respect_robots=respect_robots,
                    cookies=cookies,
                )
            params = clean_params(
                {
                    "search": search,
                    "limit": limit,
                    "sitemap": sitemap,
                    "include_subdomains": include_subdomains,
                    "ignore_query_params": ignore_query_params,
                    "respect_robots": respect_robots,
                    "cookies": cookies,
                }
            )
            result = await client.map(url, **params)
            return {
                "links": [
                    {"url": link.url, "title": getattr(link, "title", None)}
                    for link in getattr(result, "links", [])
                ],
                "total": getattr(result, "total", 0),
                "sources": getattr(result, "sources", {}),
                "credits_used": getattr(result, "credits_used", 0),
            }
        except Exception as e:
            return repr(e)


class SpidraParse(_SpidraBaseTool):
    """Parse documents into clean Markdown page-by-page.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraParse

        tool = SpidraParse()
        tool.invoke({"source": "https://example.com/document.pdf"})
        ```
    """

    name: str = "spidra_parse"
    description: str = (
        "Parse a document (PDF, Word, Excel, PowerPoint, etc. via file path, URL, or file_id) "
        "into clean Markdown page-by-page. Optionally provide a 'prompt' or 'schema' for AI "
        "extraction. Input should include 'source'."
    )

    def _run(
        self,
        source: str,
        prompt: str | None = None,
        schema: dict[str, Any] | None = None,
        mode: str | None = None,
        pages: str | None = None,
        max_pages: int | None = None,
        include_pages: bool | None = True,
        page_markers: bool | None = None,
        file_name: str | None = None,
        file_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Parse document synchronously."""
        try:
            params = clean_params(
                {
                    "prompt": prompt,
                    "schema": schema,
                    "mode": mode,
                    "pages": pages,
                    "max_pages": max_pages,
                    "include_pages": include_pages,
                    "page_markers": page_markers,
                    "file_name": file_name,
                    "file_id": file_id,
                }
            )
            result = self.client.parse(source, **params)
            return {
                "job_id": getattr(result, "job_id", ""),
                "markdown": getattr(result, "markdown", "") or "",
                "total_pages": getattr(result, "total_pages", None),
                "pages": [
                    {
                        "page_number": p.page_number,
                        "markdown": p.markdown,
                        "method": getattr(p, "method", "text"),
                    }
                    for p in (getattr(result, "pages", None) or [])
                ],
                "json": getattr(result, "json", None),
                "status": getattr(result, "status", "completed"),
            }
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        source: str,
        prompt: str | None = None,
        schema: dict[str, Any] | None = None,
        mode: str | None = None,
        pages: str | None = None,
        max_pages: int | None = None,
        include_pages: bool | None = True,
        page_markers: bool | None = None,
        file_name: str | None = None,
        file_id: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Parse document asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(
                    self._run,
                    source=source,
                    prompt=prompt,
                    schema=schema,
                    mode=mode,
                    pages=pages,
                    max_pages=max_pages,
                    include_pages=include_pages,
                    page_markers=page_markers,
                    file_name=file_name,
                    file_id=file_id,
                )
            params = clean_params(
                {
                    "prompt": prompt,
                    "schema": schema,
                    "mode": mode,
                    "pages": pages,
                    "max_pages": max_pages,
                    "include_pages": include_pages,
                    "page_markers": page_markers,
                    "file_name": file_name,
                    "file_id": file_id,
                }
            )
            result = await client.parse(source, **params)
            return {
                "job_id": getattr(result, "job_id", ""),
                "markdown": getattr(result, "markdown", "") or "",
                "total_pages": getattr(result, "total_pages", None),
                "pages": [
                    {
                        "page_number": p.page_number,
                        "markdown": p.markdown,
                        "method": getattr(p, "method", "text"),
                    }
                    for p in (getattr(result, "pages", None) or [])
                ],
                "json": getattr(result, "json", None),
                "status": getattr(result, "status", "completed"),
            }
        except Exception as e:
            return repr(e)


class SpidraBulkScrape(_SpidraBaseTool):
    """Perform template-based bulk scraping with placeholder variables.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraBulkScrape

        tool = SpidraBulkScrape()
        tool.invoke({
            "template": "https://example.com/items/{category}/{id}",
            "variables": {"category": ["books", "movies"], "id": ["1", "2"]},
            "items_path": "items",
            "dedupe_by": "id",
        })
        ```
    """

    name: str = "spidra_bulk_scrape"
    description: str = (
        "Bulk scrape pages using a URL template with placeholder variables. "
        "Batches requests and removes duplicates based on a deduplication key."
    )

    def _run(
        self,
        template: str,
        variables: dict[str, list[str]],
        items_path: str,
        dedupe_by: str,
        target: int | None = None,
        max_pages: int = 1000,
        prompt: str | None = None,
        output: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Run bulk scrape synchronously."""
        try:
            scrape_kwargs = clean_params(
                {
                    "prompt": prompt,
                    "output": output,
                }
            )
            result = self.client.bulk_scrape(
                template,
                variables,
                items_path=items_path,
                dedupe_by=dedupe_by,
                target=target,
                max_pages=max_pages,
                **scrape_kwargs,
            )
            return {
                "items": getattr(result, "items", []),
                "stats": to_dict(getattr(result, "stats", None)),
            }
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        template: str,
        variables: dict[str, list[str]],
        items_path: str,
        dedupe_by: str,
        target: int | None = None,
        max_pages: int = 1000,
        prompt: str | None = None,
        output: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Run bulk scrape asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(
                    self._run,
                    template=template,
                    variables=variables,
                    items_path=items_path,
                    dedupe_by=dedupe_by,
                    target=target,
                    max_pages=max_pages,
                    prompt=prompt,
                    output=output,
                )
            scrape_kwargs = clean_params(
                {
                    "prompt": prompt,
                    "output": output,
                }
            )
            result = await client.bulk_scrape(
                template,
                variables,
                items_path=items_path,
                dedupe_by=dedupe_by,
                target=target,
                max_pages=max_pages,
                **scrape_kwargs,
            )
            return {
                "items": getattr(result, "items", []),
                "stats": to_dict(getattr(result, "stats", None)),
            }
        except Exception as e:
            return repr(e)


class SpidraRunTemplate(_SpidraBaseTool):
    """Run a ready-made Spidra template and wait for its results.

    Templates are pre-built data extraction workflows (e.g. Google Maps leads,
    Amazon products, LinkedIn profiles). Pass the template ``slug`` and the
    required ``inputs`` dict; the tool runs the job and returns the rows.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraRunTemplate

        tool = SpidraRunTemplate()
        result = tool.invoke({
            "slug": "google-maps-search",
            "inputs": {"query": "dentist in Lagos", "limit": 20},
        })
        ```
    """

    name: str = "spidra_run_template"
    description: str = (
        "Run a ready-made Spidra data-extraction template (e.g. Google Maps, Amazon, LinkedIn) "
        "and return the structured rows. Provide the template 'slug' and an 'inputs' dict with "
        "the fields the template requires."
    )

    def _run(
        self,
        slug: str,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Run template synchronously and wait for completion."""
        try:
            run = self.client.run_template(slug, inputs)
            return {
                "run_id": run.run_id,
                "status": run.status,
                "template": run.template,
                "tool": run.tool,
                "row_count": run.row_count,
                "url_count": run.url_count,
                "credits_used": run.credits_used,
                "rows": run.rows or [],
                "failures": run.failures,
                "error": run.error,
            }
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        slug: str,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Run template asynchronously and wait for completion."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(self._run, slug=slug, inputs=inputs)
            run = await client.run_template(slug, inputs)
            return {
                "run_id": run.run_id,
                "status": run.status,
                "template": run.template,
                "tool": run.tool,
                "row_count": run.row_count,
                "url_count": run.url_count,
                "credits_used": run.credits_used,
                "rows": run.rows or [],
                "failures": run.failures,
                "error": run.error,
            }
        except Exception as e:
            return repr(e)


class SpidraRunPreset(_SpidraBaseTool):
    """Run a saved Spidra preset (template preset or tool preset).

    Presets are named, reusable jobs. A template preset stores a template slug
    and its inputs; a tool preset stores the request body for one tool
    (scrape, batch, crawl, map, search, or parse).

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraRunPreset

        tool = SpidraRunPreset()
        result = tool.invoke({"preset_id": "abc123"})
        ```
    """

    name: str = "spidra_run_preset"
    description: str = (
        "Run a saved Spidra preset by its ID. A preset is a named, reusable job — either a "
        "template with saved inputs or a specific tool configuration. Provide the 'preset_id'. "
        "Optionally supply 'overrides' (a dict) to override individual inputs for this run."
    )

    def _run(
        self,
        preset_id: str,
        overrides: dict[str, Any] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Run preset synchronously."""
        try:
            response = self.client.run_preset(preset_id, overrides=overrides, wait=True)
            result: dict[str, Any] = {
                "preset_id": response.preset_id,
                "tool": response.tool,
            }
            if response.run is not None:
                run = response.run
                result["run"] = {
                    "run_id": run.run_id,
                    "status": run.status,
                    "template": run.template,
                    "tool": run.tool,
                    "row_count": run.row_count,
                    "credits_used": run.credits_used,
                    "rows": run.rows or [],
                    "failures": run.failures,
                    "error": run.error,
                }
            else:
                result["result"] = response.result
            return result
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        preset_id: str,
        overrides: dict[str, Any] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Run preset asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(
                    self._run, preset_id=preset_id, overrides=overrides
                )
            response = await client.run_preset(preset_id, overrides=overrides, wait=True)
            result: dict[str, Any] = {
                "preset_id": response.preset_id,
                "tool": response.tool,
            }
            if response.run is not None:
                run = response.run
                result["run"] = {
                    "run_id": run.run_id,
                    "status": run.status,
                    "template": run.template,
                    "tool": run.tool,
                    "row_count": run.row_count,
                    "credits_used": run.credits_used,
                    "rows": run.rows or [],
                    "failures": run.failures,
                    "error": run.error,
                }
            else:
                result["result"] = response.result
            return result
        except Exception as e:
            return repr(e)


class SpidraMonitor(_SpidraBaseTool):
    """Check a Spidra monitor for the latest changes on a schedule.

    Monitors run data sources (templates, presets, or URL lists) on a schedule
    and alert when something changes. This tool retrieves the most recent check
    result for a given monitor ID so an agent can react to the detected changes.

    Setup:
        Install ``langchain-spidra`` and set ``SPIDRA_API_KEY``.

    Instantiation:
        ```python
        from langchain_spidra import SpidraMonitor

        tool = SpidraMonitor()
        result = tool.invoke({"monitor_id": "mon_abc123"})
        ```
    """

    name: str = "spidra_monitor"
    description: str = (
        "Retrieve the latest check result for a Spidra monitor. Monitors run data sources on a "
        "schedule and detect changes (new, removed, or updated rows). Provide the 'monitor_id'. "
        "Returns the most recent check summary including what was added, changed, or removed."
    )

    def _run(
        self,
        monitor_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Fetch the latest monitor check result synchronously."""
        try:
            monitor = self.client.get_monitor(monitor_id)
            checks = self.client.list_monitor_checks(monitor_id, limit=1)
            latest = checks.checks[0] if checks.checks else None
            result: dict[str, Any] = {
                "monitor_id": monitor_id,
                "name": monitor.name,
                "status": monitor.status,
                "last_check_at": monitor.last_check_at,
                "last_change_at": monitor.last_change_at,
                "last_summary": monitor.last_summary,
            }
            if latest is not None:
                result["latest_check"] = {
                    "check_id": latest.id,
                    "trigger": latest.trigger,
                    "status": latest.status,
                    "has_changes": latest.has_changes,
                    "summary": latest.summary,
                    "credits_used": latest.credits_used,
                    "started_at": latest.started_at,
                    "finished_at": latest.finished_at,
                }
            return result
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        monitor_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any] | str:
        """Fetch the latest monitor check result asynchronously."""
        try:
            client = self.async_client
            if client is None:
                return await asyncio.to_thread(self._run, monitor_id=monitor_id)
            monitor = await client.get_monitor(monitor_id)
            checks = await client.list_monitor_checks(monitor_id, limit=1)
            latest = checks.checks[0] if checks.checks else None
            result: dict[str, Any] = {
                "monitor_id": monitor_id,
                "name": monitor.name,
                "status": monitor.status,
                "last_check_at": monitor.last_check_at,
                "last_change_at": monitor.last_change_at,
                "last_summary": monitor.last_summary,
            }
            if latest is not None:
                result["latest_check"] = {
                    "check_id": latest.id,
                    "trigger": latest.trigger,
                    "status": latest.status,
                    "has_changes": latest.has_changes,
                    "summary": latest.summary,
                    "credits_used": latest.credits_used,
                    "started_at": latest.started_at,
                    "finished_at": latest.finished_at,
                }
            return result
        except Exception as e:
            return repr(e)

