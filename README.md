# langchain-spidra

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-spidra?label=%20)](https://pypi.org/project/langchain-spidra/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-spidra)](https://opensource.org/licenses/MIT)

Official LangChain integration package for [Spidra](https://spidra.io), an AI-powered web data platform for web scraping, intelligent crawling, search, document parsing, ready-made extraction templates, and scheduled monitors.

This package provides native LangChain **Agent Tools**, **Document Loaders**, and **Retrievers** with full synchronous and asynchronous (`asyncio`) support.

---

## Features

- **Agent Tools**: 10 purpose-built tools for LangChain agents and LangGraph workflows.
- **Web & Research Search**: Search web, news, academic papers (arXiv, PubMed, bioRxiv), and developer code (GitHub).
- **AI-Powered Scraping**: Turn unstructured HTML into clean markdown or structured JSON using natural language prompts or JSON schemas.
- **Intelligent Crawling**: Traverse websites with depth control, path filters, sitemap awareness, and robots.txt compliance.
- **Site Mapping**: Discover and index all URLs across a domain via sitemaps and discovered links.
- **Document Parsing**: Extract text and tabular data page-by-page from PDFs, Word docs, Excel sheets, and PowerPoint presentations.
- **Ready-Made Templates**: Run pre-built data extraction workflows (Google Maps, Amazon, LinkedIn, Reddit, and more).
- **Saved Presets**: Execute saved template or tool configurations by preset ID with dynamic overrides.
- **Scheduled Monitors**: Query change-detection summaries and alert history from recurring monitoring jobs.
- **Document Loaders**: Load scraped pages, crawled sites, search results, or parsed files directly into LangChain `Document` objects for RAG.
- **Native Async**: Every tool and loader implements both `invoke` / `load` and `ainvoke` / `aload`.

---

## Installation

```bash
pip install -U langchain-spidra
```

### Authentication

Get your API key from the [Spidra Dashboard](https://app.spidra.io) and export it as an environment variable:

```bash
export SPIDRA_API_KEY="spd_YOUR_API_KEY"
```

Alternatively, pass `api_key` directly when instantiating any tool or loader:

```python
from langchain_spidra import SpidraSearch

tool = SpidraSearch(api_key="spd_YOUR_API_KEY")
```

---

## Tools for LangChain Agents

All tools inherit from LangChain's `BaseTool` and can be provided directly to agents, chains, or called standalone.

| Tool | Class | Description |
| :--- | :--- | :--- |
| **Search** | `SpidraSearch` | Real-time search across web, news, academic papers, and developer code. |
| **Scrape** | `SpidraScrape` | Single-page scrape with AI prompt extraction, JSON schema, or markdown. |
| **Crawl** | `SpidraCrawl` | Intelligent multi-page website crawler with depth and path filtering. |
| **Map** | `SpidraMap` | Rapid URL discovery and site mapping from sitemaps and links. |
| **Parse** | `SpidraParse` | Multi-format document parser (PDF, DOCX, XLSX, PPTX, etc.). |
| **Batch Scrape** | `SpidraBatchScrape` | Parallel scraping across a list of URLs. |
| **Bulk Scrape** | `SpidraBulkScrape` | Parameterized URL pattern scraping with variable matrices. |
| **Templates** | `SpidraRunTemplate` | Run pre-built extraction workflows (Google Maps, Amazon, LinkedIn, etc.). |
| **Presets** | `SpidraRunPreset` | Run saved named presets by ID with optional input overrides. |
| **Monitors** | `SpidraMonitor` | Inspect the latest change-detection results from a scheduled monitor. |

---

### Tool Usage Examples

#### 1. Search the Web, News & Research

```python
from langchain_spidra import SpidraSearch

search = SpidraSearch()

# Basic web search
results = search.invoke({"query": "LangChain autonomous agent architectures", "limit": 5})

# Academic research papers
papers = search.invoke({
    "query": "transformer attention mechanisms",
    "sources": ["research"],
    "limit": 5,
})
for p in papers["results"]:
    print(p["title"], p.get("authors"), p.get("doi"))
```

#### 2. Single-Page Scraping with AI Extraction

```python
from langchain_spidra import SpidraScrape

scrape = SpidraScrape()

# Extract structured data using an instruction prompt
result = scrape.invoke({
    "url": "https://example.com/pricing",
    "prompt": "Extract all plan names, monthly prices, and key features as a JSON list",
    "output": "json",
})
print(result["content"])
```

#### 3. Website Crawling

```python
from langchain_spidra import SpidraCrawl

crawl = SpidraCrawl()

pages = crawl.invoke({
    "url": "https://docs.example.com",
    "crawl_instruction": "Find all API reference and authentication guides",
    "max_pages": 15,
    "depth": 2,
    "respect_robots": True,
})

for page in pages:
    print(page["url"], page["title"])
```

#### 4. Site Architecture Mapping

```python
from langchain_spidra import SpidraMap

mapper = SpidraMap()
site_map = mapper.invoke({"url": "https://docs.example.com", "limit": 100})

print(f"Discovered {site_map['total']} URLs:")
for link in site_map["links"]:
    print(link["url"])
```

#### 5. Document Parsing (PDF, Word, Excel, PPTX)

```python
from langchain_spidra import SpidraParse

parser = SpidraParse()

doc = parser.invoke({
    "source": "https://example.com/quarterly-report.pdf",
    "pages": "1-10",
    "include_pages": True,
})

print(doc["markdown"])
for page in doc.get("pages", []):
    print(f"Page {page['page_number']}: {page['method']}")
```

#### 6. Parameterized Bulk Scraping

```python
from langchain_spidra import SpidraBulkScrape

bulk = SpidraBulkScrape()

result = bulk.invoke({
    "template": "https://example.com/catalog/{category}/{item_id}",
    "variables": {
        "category": ["books", "electronics"],
        "item_id": ["101", "102"],
    },
    "items_path": "items",
    "dedupe_by": "item_id",
})

print(f"Extracted {len(result['items'])} items")
```

#### 7. Ready-Made Templates

Run curated, pre-built extraction workflows without writing custom scrapers:

```python
from langchain_spidra import SpidraRunTemplate

template_tool = SpidraRunTemplate()

# Google Maps leads extraction
leads = template_tool.invoke({
    "slug": "google-maps-search",
    "inputs": {
        "query": "dentists in Austin, TX",
        "limit": 20,
    },
})

for shop in leads["rows"]:
    print(shop.get("title"), shop.get("phone"), shop.get("address"))
```

#### 8. Saved Presets

Run a pre-configured template or tool preset stored in your Spidra workspace:

```python
from langchain_spidra import SpidraRunPreset

preset_tool = SpidraRunPreset()

# Run a saved preset, overriding specific fields as needed
result = preset_tool.invoke({
    "preset_id": "pre_abc123",
    "overrides": {"limit": 50},
})

print(result)
```

#### 9. Scheduled Monitors

Retrieve change-detection status, added/removed elements, and summaries from scheduled monitor checks:

```python
from langchain_spidra import SpidraMonitor

monitor_tool = SpidraMonitor()

check = monitor_tool.invoke({"monitor_id": "mon_abc123"})

print("Monitor:", check["name"])
if check.get("latest_check"):
    print("Has changes:", check["latest_check"]["has_changes"])
    print("Summary:", check["latest_check"]["summary"])
```

---

### Async Execution

All tools natively support `ainvoke`:

```python
import asyncio
from langchain_spidra import SpidraSearch, SpidraRunTemplate

async def main():
    search = SpidraSearch()
    template = SpidraRunTemplate()

    # Run searches and templates concurrently
    search_task = search.ainvoke({"query": "AI agent benchmarks 2026", "limit": 5})
    template_task = template.ainvoke({
        "slug": "amazon-product-search",
        "inputs": {"query": "mechanical keyboard", "limit": 10},
    })

    search_res, template_res = await asyncio.gather(search_task, template_task)
    print("Search items:", len(search_res["results"]))
    print("Template rows:", template_res["row_count"])

asyncio.run(main())
```

---

### LangChain Agent Integration

Provide tools directly to a tool-calling LLM:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_spidra import SpidraSearch, SpidraScrape, SpidraRunTemplate

tools = [
    SpidraSearch(),
    SpidraScrape(),
    SpidraRunTemplate(),
]

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

response = llm.invoke([
    HumanMessage(content="Find the latest pricing plans for Vercel and extract them as structured data.")
])
print(response.tool_calls)
```

---

## Document Loaders

Load web and document content directly into LangChain `Document` objects for vector embeddings and RAG pipelines.

### Generic `SpidraLoader`

Supports both single-URL scrape and multi-page crawl modes:

```python
from langchain_spidra import SpidraLoader

# Scrape a single page
loader = SpidraLoader(url="https://example.com/about", mode="scrape")
docs = loader.load()

# Crawl documentation
crawler_loader = SpidraLoader(
    url="https://docs.example.com",
    mode="crawl",
    max_pages=20,
    depth=2,
    respect_robots=True,
)
docs = crawler_loader.load()
```

### Specialized Loaders

```python
from langchain_spidra import (
    SpidraSearchLoader,
    SpidraParseLoader,
    SpidraMapLoader,
)

# 1. Search Loader — loads search results as Documents with metadata
search_loader = SpidraSearchLoader("autonomous agent memory architectures", limit=5)
search_docs = search_loader.load()

# 2. Document Parser Loader — parses PDF, Word, Excel page-by-page
pdf_loader = SpidraParseLoader("https://example.com/whitepaper.pdf")
pdf_docs = pdf_loader.load()
for page in pdf_docs:
    print(f"Page {page.metadata['page_number']}: {page.page_content[:120]}...")

# 3. Site Map Loader — loads discovered URLs as Document objects
map_loader = SpidraMapLoader("https://docs.example.com")
map_docs = map_loader.load()
```

---

## Retriever for RAG

Use `SpidraRetriever` inside LangChain Expression Language (LCEL) pipelines:

```python
from langchain_spidra import SpidraRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

retriever = SpidraRetriever(
    sources=["web", "news"],
    limit=5,
)

prompt = ChatPromptTemplate.from_template("""Answer the question based on the context:
Context: {context}
Question: {question}""")

chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

answer = chain.invoke("What are the key announcements in the latest AI research papers?")
print(answer)
```

---

## Configuration & Options

All tools and loaders accept common client configuration parameters:

| Parameter | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `api_key` | `SPIDRA_API_KEY` | `None` | Spidra API key. |
| `base_url` | `SPIDRA_BASE_URL` | `https://api.spidra.io/v1` | API endpoint base URL. |
| `timeout` | `SPIDRA_TIMEOUT` | `120.0` | Request timeout in seconds. |
| `max_retries` | `SPIDRA_MAX_RETRIES` | `3` | Maximum automatic retries on network failures. |

---

## Links & Resources

- **Documentation**: [docs.spidra.io](https://docs.spidra.io)
- **Spidra Platform**: [spidra.io](https://spidra.io)
- **Dashboard & API Keys**: [app.spidra.io](https://app.spidra.io)
- **Python SDK**: [pypi.org/project/spidra](https://pypi.org/project/spidra/)
- **LangChain Integration**: [pypi.org/project/langchain-spidra](https://pypi.org/project/langchain-spidra/)
- **Issue Tracker**: [GitHub Issues](https://github.com/spidra-io/spidra-langchain/issues)
