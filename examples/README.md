# Spidra LangChain Examples

This directory contains premium examples of how to integrate Spidra with LangChain for powerful web scraping, crawling, and AI tasks.

## Setup Instructions

### 1. Install Dependencies

Ensure you have a virtual environment set up and active. Install the required libraries:

```bash
pip install langchain-spidra langchain-openai langchain-text-splitters pydantic
```

### 2. Export API Keys

Export your Spidra and OpenAI API keys to your environment:

```bash
export SPIDRA_API_KEY="spd_your_api_key_here"
export OPENAI_API_KEY="sk-proj-your_openai_key_here"
```

---

## Running the Examples

### 1. Scrape & Summarize (Scrape + Chat)

Scrapes a single webpage using `SpidraLoader` and generates a 2-3 sentence summary using `ChatOpenAI`.

```bash
python examples/scrape_and_chat.py
```

### 2. Structured Extraction

Scrapes a webpage and extracts typed structured information into a Pydantic schema using LangChain's structured output.

```bash
python examples/structured_extraction.py
```

### 3. Tool-Calling Agent

Binds `SpidraScrape` as a tool to a LangChain agent, enabling the model to dynamically request scraping a website to answer user questions.

```bash
python examples/tool_calling_agent.py
```

### 4. Crawl & RAG Pipeline

Crawls multiple pages of a website using `SpidraLoader` in `crawl` mode, splits the content, indexes it in an in-memory vector store, and performs retrieval Q&A.

```bash
python examples/rag_pipeline.py
```
