from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langchain_spidra import SpidraLoader

# Ensure you set the required environment variables:
# export SPIDRA_API_KEY="your-spidra-key"
# export OPENAI_API_KEY="your-openai-key"


# 1. Define the target structured schema using Pydantic
class CompanyInfo(BaseModel):
    name: str = Field(description="The formal name of the company or organization")
    domain: str = Field(description="The primary business domain or industry")
    summary: str = Field(description="A concise summary of what the company does")
    main_features: list[str] = Field(
        description="List of primary features, offerings or products mentioned"
    )


def main() -> None:
    # 2. Scrape the website content
    url = "https://example.com"
    print(f"Scraping {url} via Spidra...")

    loader = SpidraLoader(url=url, mode="scrape")
    docs = loader.load()
    if not docs:
        print("No content scraped.")
        return

    scraped_text = docs[0].page_content
    print(f"Loaded {len(scraped_text)} characters of page content.")

    # 3. Setup LLM with Structured Output binding
    print("\nInitializing LLM and binding Pydantic structured output schema...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(CompanyInfo)

    # 4. Invoke LLM to perform extraction
    print("Extracting structured fields...")
    prompt = (
        "Analyze the following web page content and extract the structured company details.\n\n"
        f"Content:\n{scraped_text}"
    )

    company_details = structured_llm.invoke(prompt)

    # 5. Display the structured typed output
    assert isinstance(company_details, CompanyInfo)
    print("\n--- Extracted Structured Data ---")
    print(f"Name:        {company_details.name}")
    print(f"Domain:      {company_details.domain}")
    print(f"Summary:     {company_details.summary}")
    print(f"Features:    {company_details.main_features}")
    print("---------------------------------")


if __name__ == "__main__":
    main()
