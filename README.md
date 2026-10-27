# langchain-spidra

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-spidra?label=%20)](https://pypi.org/project/langchain-spidra/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-spidra)](https://opensource.org/licenses/MIT)

This package contains the official LangChain integration with [Spidra](https://spidra.io), an AI-powered web scraping, search, crawling, and document parsing platform.

---

## What's New in v0.3.0

`langchain-spidra` v0.3.0 brings full parity with the `spidra` Python SDK v0.6.0:

| Feature | Description | Tools |
| :--- | :--- | :--- |
| **Templates** | Run 100+ ready-made data-extraction workflows (Google Maps, Amazon, LinkedIn, Reddit, etc.) | `SpidraRunTemplate` |
| **Presets** | Run a saved named job (template preset or tool preset) by its ID | `SpidraRunPreset` |
| **Monitors** | Fetch the latest change-detection check from a scheduled monitor | `SpidraMonitor` |

All three new tools support both sync (`invoke`) and async (`ainvoke`) execution.

---

## What's New in v0.2.0

`langchain-spidra` v0.2.0 brought full parity with the `spidra` Python SDK v0.5.1+:

| Feature | Description | Tools / Loaders |
| :--- | :--- | :--- |
| **Web & Research Search** | Search web, news, academic papers, and developer code in real-time | `SpidraSearch`, `SpidraSearchLoader`, `SpidraRetriever` |
| **Site Mapping** | Discover all URLs and sub-pages on any domain via sitemaps & page links | `SpidraMap`, `SpidraMapLoader` |
| **Document Parsing** | Parse PDF, Word (DOCX), Excel (XLSX), PPTX into markdown page-by-page | `SpidraParse`, `SpidraParseLoader` |
| **Bulk Templated Scraping** | Scrape parameterized URL patterns with variable matrices & deduplication | `SpidraBulkScrape` |
| **LangChain Retriever** | Seamless RAG retrieval with search query expansion and metadata | `SpidraRetriever` |
| **Full Async Support** | All tools and retrievers implement native async (`ainvoke` / `_arun`) | All Tools & Retrievers |
| **Advanced Crawl Controls** | Control crawl `depth`, filter `parent_url`, `respect_robots`, `use_sitemap` | `SpidraCrawl`, `SpidraLoader` |

---

## Quick Install

```bash
pip install -U langchain-spidra
```

Get an API key from [spidra.io](https://spidra.io) and set it as an environment variable (or pass `api_key=...` directly):

```bash
export SPIDRA_API_KEY="spd-your-api-key"
```

---

## Agent Tools

Every Spidra capability is available as a LangChain `BaseTool` that can be bound to any LangChain agent, LangGraph workflow, or invoked directly. All tools support both synchronous (`invoke`) and asynchronous (`ainvoke`) execution.

### Available Tools

- `SpidraSearch`: Real-time search across web, news, academic, and code sources.
- `SpidraScrape`: Single page scraping with AI extraction and structured schema.
- `SpidraCrawl`: Full website crawling with depth and sitemap control.
- `SpidraMap`: Rapid URL discovery and site mapping.
- `SpidraParse`: Multi-format document parser (PDF, DOCX, XLSX, etc.).
- `SpidraBatchScrape`: Parallel scraping across explicit URL lists.
- `SpidraBulkScrape`: URL template pattern scraping with variable matrices.
- `SpidraRunTemplate`: Run a ready-made data-extraction template (Google Maps, Amazon, LinkedIn, etc.).
- `SpidraRunPreset`: Run a saved named preset by its ID.
- `SpidraMonitor`: Fetch the latest change-detection check from a scheduled monitor.

### Core Tools Usage Example

```python
from langchain_spidra import (
    SpidraSearch,
    SpidraScrape,
    SpidraCrawl,
    SpidraMap,
    SpidraParse,
    SpidraBulkScrape,
    SpidraBatchScrape,
)

# 1. Search the web
search_tool = SpidraSearch()
results = search_tool.invoke({"query": "LangChain agent workflows", "limit": 5})

# 2. Scrape with AI extraction
scrape_tool = SpidraScrape()
data = scrape_tool.invoke({
    "url": "https://example.com/pricing",
    "prompt": "Extract all tier names and prices as a JSON table",
    "scrape_mode": "markdown",
})

# 3. Crawl a website with depth control
crawl_tool = SpidraCrawl()
pages = crawl_tool.invoke({
    "url": "https://docs.example.com",
    "max_pages": 15,
    "depth": 2,
    "respect_robots": True,
})

# 4. Map site architecture
map_tool = SpidraMap()
links = map_tool.invoke({"url": "https://docs.example.com"})

# 5. Parse a PDF or document into markdown
parse_tool = SpidraParse()
doc = parse_tool.invoke({"source": "https://example.com/whitepaper.pdf"})

# 6. Bulk scrape with URL templates
bulk_tool = SpidraBulkScrape()
bulk = bulk_tool.invoke({
    "template": "https://example.com/catalog/{category}/{item_id}",
    "variables": {"category": ["ai", "cloud"], "item_id": ["1", "2"]},
    "dedupe_by": "item_id",
})
```

### Templates, Presets & Monitors (v0.3.0)

```python
from langchain_spidra import SpidraRunTemplate, SpidraRunPreset, SpidraMonitor

# Run a ready-made template — returns structured rows
template_tool = SpidraRunTemplate()
result = template_tool.invoke({
    "slug": "google-maps-search",
    "inputs": {
        "query": "dentist in Lagos",
        "limit": 20,
    },
})
# result["rows"] — list of place dicts (name, address, rating, phone, website, …)
# result["status"] — "completed" | "partial" | "failed"
# result["credits_used"] — credits charged for this run

# Run a saved preset by its ID
preset_tool = SpidraRunPreset()
preset_result = preset_tool.invoke({
    "preset_id": "abc123",
    # Optional: override specific inputs for this run only
    "overrides": {"limit": 50},
})
# preset_result["run"] — TemplateRun fields when it's a template preset
# preset_result["result"] — raw result dict when it's a tool preset

# Check the latest change-detection result from a monitor
monitor_tool = SpidraMonitor()
check = monitor_tool.invoke({"monitor_id": "mon_abc123"})
# check["has_changes"] — True when something changed since the last check
# check["latest_check"]["summary"] — AI-generated summary of what changed
# check["last_change_at"] — ISO timestamp of the last detected change
```

### Async Agent Example

```python
import asyncio
from langchain_spidra import SpidraSearch, SpidraScrape, SpidraRunTemplate

async def main():
    # Search
    search = SpidraSearch()
    results = await search.ainvoke({"query": "Autonomous coding agents", "limit": 3})
    print(results)

    # Run a template asynchronously
    template = SpidraRunTemplate()
    run = await template.ainvoke({
        "slug": "amazon-product-search",
        "inputs": {"query": "wireless headphones", "limit": 10},
    })
    print(f"Got {run['row_count']} products")

asyncio.run(main())
```

---

## Document Loaders

Load web and file content directly as LangChain `Document` objects for vector stores, embeddings, and RAG pipelines.

### Generic `SpidraLoader`

```python
from langchain_spidra import SpidraLoader

# Scrape
loader = SpidraLoader(url="https://example.com", mode="scrape")
docs = loader.load()

# Crawl documentation
loader = SpidraLoader(
    url="https://docs.example.com",
    mode="crawl",
    max_pages=20,
    depth=2,
)
docs = loader.load()
```

### Specialized Loaders

```python
from langchain_spidra import (
    SpidraSearchLoader,
    SpidraParseLoader,
    SpidraMapLoader,
)

# Search loader
search_loader = SpidraSearchLoader("latest AI advancements", limit=5)
search_docs = search_loader.load()

# Document parser loader (PDF, DOCX, XLSX, PPTX)
pdf_loader = SpidraParseLoader("https://example.com/financials.pdf")
pdf_pages = pdf_loader.load()
for page in pdf_pages:
    print(f"Page {page.metadata.get('page_number')}: {page.page_content[:100]}...")

# Site mapper loader
map_loader = SpidraMapLoader("https://docs.example.com")
url_docs = map_loader.load()
```

---

## Retriever (RAG)

Use `SpidraRetriever` directly within LangChain Expression Language (LCEL) chains:

```python
from langchain_spidra import SpidraRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

retriever = SpidraRetriever(
    sources=["web", "news"],
    limit=5,
)

# Direct retrieval
docs = retriever.invoke("What are the best practices for AI agent memory?")

# Async retrieval
# docs = await retriever.ainvoke("What are the best practices for AI agent memory?")
```

---

## Documentation & Support

- **Documentation**: [docs.spidra.io](https://docs.spidra.io)
- **Website**: [spidra.io](https://spidra.io)
- **Issues**: [GitHub Issues](https://github.com/spidra-io/spidra-langchain/issues)
