# langchain-spidra

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-spidra?label=%20)](https://pypi.org/project/langchain-spidra/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-spidra)](https://opensource.org/licenses/MIT)

This package contains the LangChain integration with [Spidra](https://spidra.io),
an AI-powered web scraping and crawling API. It lets you scrape, crawl, and
batch-scrape the web — as a document loader for RAG pipelines or as agent tools.

## Quick Install

```bash
pip install langchain-spidra
```

Get an API key from [spidra.io](https://spidra.io) and set it as the
`SPIDRA_API_KEY` environment variable (or pass `api_key=...`).

```bash
export SPIDRA_API_KEY="spd-your-api-key"
```

## Document loader

`SpidraLoader` loads web content as LangChain `Document`s. Pick a `mode`:
`scrape` (one page), `crawl` (a whole site), or `batch` (many URLs in parallel).

```python
from langchain_spidra import SpidraLoader

# Scrape a single page
loader = SpidraLoader(url="https://example.com", mode="scrape")
docs = loader.load()
print(docs[0].page_content[:200])
print(docs[0].metadata)

# Crawl an entire docs site
loader = SpidraLoader(url="https://docs.example.com", mode="crawl")
docs = loader.load()

# Batch scrape many URLs
loader = SpidraLoader(
    urls=["https://a.com", "https://b.com", "https://c.com"],
    mode="batch",
)
docs = loader.load()
```

## Tools

Each Spidra capability is also available as a `BaseTool` you can bind to an
agent:

```python
from langchain_spidra import SpidraScrape, SpidraCrawl, SpidraBatchScrape

scrape = SpidraScrape()
result = scrape.invoke({"url": "https://example.com"})
print(result["content"])

crawl = SpidraCrawl()
pages = crawl.invoke({"url": "https://docs.example.com", "max_pages": 20})
for page in pages:
    print(page["url"], page["data"][:100])

batch = SpidraBatchScrape()
items = batch.invoke({"urls": ["https://a.com", "https://b.com"]})
```

## Documentation

- Spidra docs: [docs.spidra.io](https://docs.spidra.io)
- Spidra homepage: [spidra.io](https://spidra.io)
