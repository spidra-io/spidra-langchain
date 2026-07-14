from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_spidra import SpidraLoader

# Ensure you set the required environment variables:
# export SPIDRA_API_KEY="your-spidra-key"
# export OPENAI_API_KEY="your-openai-key"


def main() -> None:
    # 1. Crawl a documentation site using Spidra
    base_url = "https://docs.spidra.io"
    print(f"Crawling documentation site {base_url}...")

    # We set mode="crawl" and limit to max_pages=3 for this demonstration
    loader = SpidraLoader(
        url=base_url,
        mode="crawl",
        params={
            "max_pages": 3,
            "max_depth": 2,
        },
    )

    docs = loader.load()
    print(f"Successfully loaded {len(docs)} documents.")
    for doc in docs:
        src = doc.metadata.get("source")
        title = doc.metadata.get("title", "N/A")
        print(f" - Page URL: {src} (Title: {title})")

    # 2. Chunk documents for vector storage
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text splits.")

    # 3. Initialize Embeddings and Vector Store
    # We use InMemoryVectorStore from langchain-core to run without database setup.
    print("Generating embeddings and indexing in vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore.from_documents(splits, embeddings)

    # 4. Ask a question and retrieve context
    query = "How do I authenticate with Spidra?"
    print(f"\nQuerying the vector store: '{query}'")

    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(query)

    print(f"Retrieved {len(retrieved_docs)} relevant segments:")
    for doc in retrieved_docs:
        print(f" - Context source: {doc.metadata.get('source')}")

    # 5. Generate answer using retrieved context
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = (
        "You are a helpful support assistant. Answer the user's question based strictly on the "
        "provided context. If you do not know the answer, say that you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    print("\nGenerating final response...")
    response = llm.invoke(prompt)
    print("\n--- Answer ---")
    print(response.content)
    print("--------------")


if __name__ == "__main__":
    main()
