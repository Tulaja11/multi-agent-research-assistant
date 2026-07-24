import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Initialize Tavily search client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def researcher_agent(state: dict) -> dict:
    """
    Agent 2: Takes each sub-question from the planner
    and searches the web for relevant information.
    """
    sub_questions = state["sub_questions"]
    search_results = []

    print(f"\n[Researcher] Searching for {len(sub_questions)} sub-questions...")

    for i, question in enumerate(sub_questions, 1):
        print(f"\n[Researcher] Searching ({i}/{len(sub_questions)}): {question}")

        try:
            # Search the web using Tavily
            result = tavily.search(
                query=question,
                max_results=3,          # top 3 results per question
                search_depth="basic"    # basic = faster, advanced = deeper
            )

            # Extract just the useful parts
            sources = []
            for r in result.get("results", []):
                sources.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")
                })

            search_results.append({
                "question": question,
                "sources": sources
            })

            print(f"[Researcher] Found {len(sources)} sources")

        except Exception as e:
            print(f"[Researcher] Search failed for question {i}: {e}")
            search_results.append({
                "question": question,
                "sources": []
            })

    print(f"\n[Researcher] Done. Collected results for {len(search_results)} questions.")

    # Write output back to state
    return {"search_results": search_results}


# Quick test
if __name__ == "__main__":
    # Simulate state coming from planner
    test_state = {
        "sub_questions": [
            "How is AI improving disease detection in Indian hospitals?",
            "What are the ethical concerns of AI in Indian healthcare?"
        ]
    }
    result = researcher_agent(test_state)

    print("\n--- SEARCH RESULTS PREVIEW ---")
    for item in result["search_results"]:
        print(f"\nQ: {item['question']}")
        print(f"Sources found: {len(item['sources'])}")
        for s in item['sources']:
            print(f"  - {s['title']}")
            print(f"    {s['url']}")