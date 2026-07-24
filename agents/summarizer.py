import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


def summarizer_agent(state: dict) -> dict:
    """
    Agent 3: Takes raw search results for each sub-question
    and summarizes them into clean, concise paragraphs.
    """
    search_results = state["search_results"]
    summaries = []

    print(f"\n[Summarizer] Summarizing {len(search_results)} sets of search results...")

    for i, item in enumerate(search_results, 1):
        question = item["question"]
        sources = item["sources"]

        print(f"\n[Summarizer] Summarizing ({i}/{len(search_results)}): {question}")

        if not sources:
            summaries.append({
                "question": question,
                "summary": "No information found for this question."
            })
            continue

        # Combine all source content into one block
        combined_content = ""
        for s in sources:
            combined_content += f"\nSource: {s['title']}\n{s['content']}\n"

        prompt = f"""
You are a research summarizer. Given a question and raw content from 
multiple web sources, write a clear, concise summary (3-4 sentences) 
that directly answers the question.

Rules:
- Be factual and specific
- Don't repeat information
- Write in plain, professional English
- Do NOT include source names or URLs in your summary

Question: {question}

Raw content from sources:
{combined_content}

Summary:
"""

        response = llm.invoke([HumanMessage(content=prompt)])

        # Handle both string and list response formats
        content = response.content
        if isinstance(content, list):
            summary = " ".join([
                c if isinstance(c, str) else c.get("text", "")
                for c in content
            ]).strip()
        else:
            summary = content.strip()

        summaries.append({
            "question": question,
            "summary": summary
        })

        print(f"[Summarizer] Done. Summary length: {len(summary)} characters")

    print(f"\n[Summarizer] All {len(summaries)} summaries complete.")

    return {"summaries": summaries}


# Quick test
if __name__ == "__main__":
    test_state = {
        "search_results": [
            {
                "question": "How is AI improving disease detection in Indian hospitals?",
                "sources": [
                    {
                        "title": "AI in Indian Healthcare",
                        "url": "https://example.com",
                        "content": """AI-powered diagnostic tools are being deployed across major 
                        Indian hospitals to detect diseases like tuberculosis, diabetic retinopathy, 
                        and cancer at early stages. Companies like Niramai and Qure.ai have developed 
                        AI solutions specifically for Indian healthcare challenges. These tools have 
                        shown 90%+ accuracy in early detection, significantly reducing misdiagnosis 
                        rates in rural areas where specialist doctors are scarce."""
                    }
                ]
            }
        ]
    }

    result = summarizer_agent(test_state)

    print("\n--- SUMMARY OUTPUT ---")
    for item in result["summaries"]:
        print(f"\nQ: {item['question']}")
        print(f"Summary: {item['summary']}")