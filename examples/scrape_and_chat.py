from langchain_openai import ChatOpenAI

from langchain_spidra import SpidraLoader

# Ensure you set the required environment variables:
# export SPIDRA_API_KEY="your-spidra-key"
# export OPENAI_API_KEY="your-openai-key"


def main() -> None:
    # 1. Initialize the SpidraLoader to scrape a website
    # SpidraLoader will automatically read SPIDRA_API_KEY from the environment
    url = "https://example.com"
    print(f"Scraping {url} via Spidra...")

    loader = SpidraLoader(url=url, mode="scrape")

    # Load the page content as LangChain Documents
    docs = loader.load()

    if not docs:
        print("No documents loaded.")
        return

    doc = docs[0]
    print(f"Scraped content length: {len(doc.page_content)} characters")
    print(f"Metadata: {doc.metadata}")
    print("\n--- Document Snippet ---")
    print(doc.page_content[:200])
    print("------------------------\n")

    # 2. Use ChatOpenAI to process the content
    print("Initializing LLM client and requesting summary...")
    chat = ChatOpenAI(model="gpt-4o-mini")

    summary_prompt = f"Summarize the following content in 2-3 sentences:\n\n{doc.page_content}"
    response = chat.invoke(summary_prompt)

    print("\n--- Summary ---")
    print(response.content)
    print("---------------")


if __name__ == "__main__":
    main()
