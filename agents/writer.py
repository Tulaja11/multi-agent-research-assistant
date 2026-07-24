import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


def writer_agent(state: dict) -> dict:
    """
    Agent 4: Takes all summaries and writes a complete,
    structured research report on the topic.
    """
    topic = state["topic"]
    summaries = state["summaries"]

    print(f"\n[Writer] Writing final report on: {topic}")

    # Build the summaries block
    summaries_block = ""
    for i, item in enumerate(summaries, 1):
        summaries_block += f"\nSection {i}: {item['question']}\n{item['summary']}\n"

    prompt = f"""
You are an expert research writer. Using the research summaries below,
write a comprehensive, well-structured research report.

Report structure:
1. Title
2. Executive Summary (2-3 sentences overview)
3. Key Findings (one section per summary, with a clear heading)
4. Conclusion (3-4 sentences wrapping up insights)

Rules:
- Write in clear, professional English
- Each section should flow naturally
- Be specific and factual
- Do not add information not present in the summaries

Topic: {topic}

Research Summaries:
{summaries_block}

Report:
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Handle both string and list response formats
    content = response.content
    if isinstance(content, list):
        final_report = " ".join([
            c if isinstance(c, str) else c.get("text", "")
            for c in content
        ]).strip()
    else:
        final_report = content.strip()

    print(f"[Writer] Report generated. Length: {len(final_report)} characters")

    return {"final_report": final_report}


# Quick test
if __name__ == "__main__":
    test_state = {
        "topic": "Impact of Artificial Intelligence on healthcare in India",
        "summaries": [
            {
                "question": "How is AI improving disease detection in Indian hospitals?",
                "summary": """AI-powered diagnostic tools are being deployed across Indian 
                hospitals to detect diseases like tuberculosis, cancer, and diabetic 
                retinopathy with over 90% accuracy. Companies like Niramai and Qure.ai 
                have developed specialized solutions that bridge the specialist gap in 
                rural areas by providing reliable early screenings."""
            },
            {
                "question": "What are the ethical concerns of AI in Indian healthcare?",
                "summary": """Key ethical concerns include data privacy, algorithmic bias 
                against rural and lower-income populations, and lack of regulatory 
                frameworks. The Indian government has begun addressing these through 
                draft AI governance policies, but implementation remains inconsistent 
                across states."""
            }
        ]
    }

    result = writer_agent(test_state)

    print("\n--- FINAL REPORT ---")
    print(result["final_report"])