# langchain-spidra

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-spidra?label=%20)](https://pypi.org/project/langchain-spidra/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-spidra)](https://opensource.org/licenses/MIT)

This package contains the official LangChain integration with [Spidra](https://spidra.io),
an AI-powered web scraping, search, crawling, and document parsing API. It provides:
- **Document Loaders** for scraping, crawling, searching, site mapping, and parsing documents (PDF, Word, Excel, etc.) into RAG pipelines.
- **Retriever** (`SpidraRetriever`) for semantic web search RAG with optional markdown enrichment.
- **Agent Tools** (`SpidraSearch`, `SpidraScrape`, `SpidraCrawl`, `SpidraBatchScrape`, `SpidraMap`, `SpidraParse`, `SpidraBulkScrape`) with full sync and async support.

## Quick Install

```bash
pip install -U langchain-spidra
```

Get an API key from [spidra.io](https://spidra.io) and set it as the
`SPIDRA_API_KEY` environment variable (or pass `api_key=...`).

```bash
export SPIDRA_API_KEY="spd-your-api-key"
```

---

## Document Loaders

`SpidraLoader` loads web and document content as LangChain `Document`s. Pick a `mode`:
- `scrape`: Scrape single or up to 3 pages
- `crawl`: Crawl an entire website (with `depth`, `parent_url`, `respect_robots`, `use_sitemap`)
- `batch`: Batch-scrape multiple URLs in parallel
- `search`: Search the web, news, academic research, or developer sources and load results as documents
- `map`: Discover all URLs and sub-pages on a website from sitemaps and page links
- `parse`: Parse documents (PDF, Word, Excel, PPTX, etc.) into clean Markdown page-by-page

```python
from langchain_spidra import (
    SpidraLoader,
    SpidraSearchLoader,
    SpidraParseLoader,
    SpidraMapLoader,
)

# 1. Scrape a single page
loader = SpidraLoader(url="https://example.com", mode="scrape")
docs = loader.load()

# 2. Crawl an entire documentation site
loader = SpidraLoader(url="https://docs.example.com", mode="crawl")
docs = loader.load()

# 3. Parse a PDF into markdown pages
loader = SpidraParseLoader("path/to/annual-report.pdf")
docs = loader.load()
for doc in docs:
    print(f"Page {doc.metadata['page_number']}:", doc.page_content[:100])

# 4. Search and load results as documents
loader = SpidraSearchLoader("latest artificial intelligence breakthroughs", limit=5)
docs = loader.load()

# 5. Map all URLs on a domain
loader = SpidraMapLoader("https://docs.example.com")
docs = loader.load()
```

---

## Retriever (RAG)

Use `SpidraRetriever` to retrieve relevant web and research documents for RAG pipelines and question-answering chains:

```python
from langchain_spidra import SpidraRetriever

retriever = SpidraRetriever(
    sources=["web", "news"],
    limit=5,
)

docs = retriever.invoke("What are the latest AI agents?")
for doc in docs:
    print(doc.metadata["title"])
    print(doc.page_content[:200])
```

---

## Agent Tools

Every Spidra capability is available as a `BaseTool` that can be bound to any LangChain agent or LangGraph workflow (supports both sync `invoke()` and async `ainvoke()`):

```python
from langchain_spidra import (
    SpidraSearch,
    SpidraScrape,
    SpidraCrawl,
    SpidraBatchScrape,
    SpidraMap,
    SpidraParse,
    SpidraBulkScrape,
)

# Search
search_tool = SpidraSearch()
results = search_tool.invoke({"query": "LangChain tutorial", "limit": 5})

# Scrape with optional AI extraction
scrape_tool = SpidraScrape()
data = scrape_tool.invoke({
    "url": "https://example.com",
    "prompt": "Extract the company mission statement"
})

# Crawl site
crawl_tool = SpidraCrawl()
pages = crawl_tool.invoke({"url": "https://docs.example.com", "max_pages": 10})

# Map site structure
map_tool = SpidraMap()
links = map_tool.invoke({"url": "https://docs.example.com"})

# Parse PDF or Office file
parse_tool = SpidraParse()
doc = parse_tool.invoke({"source": "https://example.com/spec.pdf"})

# Bulk scrape template
bulk_tool = SpidraBulkScrape()
bulk = bulk_tool.invoke({
    "template": "https://example.com/products/{category}/{id}",
    "variables": {"category": ["books", "electronics"], "id": ["1", "2"]},
    "items_path": "items",
    "dedupe_by": "id",
})
```

---

## Documentation

- Spidra docs: [docs.spidra.io](https://docs.spidra.io)
- Spidra homepage: [spidra.io](https://spidra.io)
