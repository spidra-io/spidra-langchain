"""Spidra Search retriever for LangChain."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr, model_validator
from spidra import AsyncSpidra, Spidra

from langchain_spidra._utilities import clean_params, initialize_client


class SpidraRetriever(BaseRetriever):
    """Spidra Search retriever integration.

    Setup:
        Install ``langchain-spidra`` and set the environment variable
        ``SPIDRA_API_KEY``.

        ```bash
        pip install -U langchain-spidra
        export SPIDRA_API_KEY="your-api-key"
        ```

    Instantiate:
        ```python
        from langchain_spidra import SpidraRetriever

        retriever = SpidraRetriever(limit=5)
        docs = retriever.invoke("latest artificial intelligence breakthroughs")
        ```
    """

    client: Spidra = Field(default=None)  # type: ignore[assignment]
    async_client: AsyncSpidra = Field(default=None)  # type: ignore[assignment]
    api_key: SecretStr = Field(default=SecretStr(""))
    api_url: str | None = None

    sources: list[str] | None = None
    limit: int = 10
    page: int | None = None
    country: str | None = None
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    time_range: str | None = None
    filetype: str | None = None
    scrape_options: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        return initialize_client(values)

    def _build_params(self) -> dict[str, Any]:
        return clean_params(
            {
                "sources": self.sources,
                "limit": self.limit,
                "page": self.page,
                "country": self.country,
                "include_domains": self.include_domains,
                "exclude_domains": self.exclude_domains,
                "time_range": self.time_range,
                "filetype": self.filetype,
                "scrape_options": self.scrape_options,
            }
        )

    def _extract_documents(self, result: Any) -> list[Document]:
        documents: list[Document] = []
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
                    documents.append(Document(page_content=str(content), metadata=metadata))
        return documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        params = self._build_params()
        result = self.client.search(query, **params)
        return self._extract_documents(result)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        params = self._build_params()
        client = self.async_client
        if client is None:
            return await asyncio.to_thread(self._get_relevant_documents, query)
        result = await client.search(query, **params)
        return self._extract_documents(result)
