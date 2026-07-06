# langchain-spidra

LangChain document loader for the [Spidra](https://spidra.io) web scraping API.

## Installation

```bash
pip install langchain-spidra
```

## Usage

### Scrape a single URL (raw markdown — best for RAG)

```python
from langchain_spidra import SpidraLoader

loader = SpidraLoader("https://example.com", api_key="spd_...")
docs = loader.load()
```

### Scrape with AI extraction

```python
loader = SpidraLoader(
    "https://example.com",
    api_key="spd_...",
    params={"prompt": "Extract product names and prices", "output": "json"},
)
docs = loader.load()
```

### Crawl an entire site

```python
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
docs = loader.load()
```

### Batch scrape many URLs

```python
loader = SpidraLoader(
    urls=["https://a.com", "https://b.com", "https://c.com"],
    mode="batch",
    api_key="spd_...",
)
docs = loader.load()
```

### Use with a vector store

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

loader = SpidraLoader("https://docs.example.com", mode="crawl", api_key="spd_...")
docs = loader.load()

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
```

## Environment variable

Set `SPIDRA_API_KEY` to avoid passing `api_key` explicitly:

```bash
export SPIDRA_API_KEY=spd_...
```
