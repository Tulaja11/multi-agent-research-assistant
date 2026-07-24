import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",   # Change if you're using another model
    google_api_key=os.getenv("GEMINI_API_KEY")
)


def planner_agent(state: dict) -> dict:
    """
    Agent 1: Takes a research topic and breaks it down into
    4 focused sub-questions.
    """
    topic = state["topic"]

    prompt = f"""
You are a research planner.

Given the research topic below, generate exactly 4 specific and searchable
sub-questions.

Rules:
- Return ONLY a numbered list.
- No explanations.
- Each question should cover a different aspect.

Topic:
{topic}
"""

    # Call Gemini
    response = llm.invoke([HumanMessage(content=prompt)])

    # Safely extract text from response
    content = response.content

    if isinstance(content, str):
        raw_output = content.strip()

    elif isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))

        raw_output = "\n".join(text_parts).strip()

    else:
        raw_output = str(content).strip()

    # Parse numbered list
    sub_questions = []

    for line in raw_output.splitlines():
        line = line.strip()

        match = re.match(r'^\d+[.)]\s*(.*)', line)
        if match:
            sub_questions.append(match.group(1))

    print("\n========== PLANNER ==========")
    print("Topic:", topic)
    print("\nGenerated Sub-Questions:")

    for i, q in enumerate(sub_questions, 1):
        print(f"{i}. {q}")

    return {"sub_questions": sub_questions}


# Test
if __name__ == "__main__":
    test_state = {
        "topic": "Impact of Artificial Intelligence on healthcare in India"
    }

    result = planner_agent(test_state)

    print("\nReturned List:")
    print(result["sub_questions"])