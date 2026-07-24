import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.summarizer import summarizer_agent
from agents.writer import writer_agent


# Define the shared state structure
# This is the "shared notebook" all agents read from and write to
class ResearchState(TypedDict):
    topic: str                        # Input from user
    sub_questions: List[str]          # Planner fills this
    search_results: List[Dict]        # Researcher fills this
    summaries: List[Dict]             # Summarizer fills this
    final_report: str                 # Writer fills this


def build_research_graph():
    """
    Builds and compiles the multi-agent research pipeline.
    Order: Planner → Researcher → Summarizer → Writer
    """
    # Create the graph with our state structure
    graph = StateGraph(ResearchState)

    # Add each agent as a node
    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("writer", writer_agent)

    # Define the flow: who runs after who
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "summarizer")
    graph.add_edge("summarizer", "writer")
    graph.add_edge("writer", END)

    # Compile and return the runnable graph
    return graph.compile()


# Quick test - runs the full pipeline end to end
if __name__ == "__main__":
    print("Building research graph...")
    research_graph = build_research_graph()

    print("Running full pipeline...\n")
    print("=" * 60)

    # Initial state - only topic is provided, agents fill the rest
    initial_state = {
        "topic": "Impact of Artificial Intelligence on healthcare in India",
        "sub_questions": [],
        "search_results": [],
        "summaries": [],
        "final_report": ""
    }

    # Run the full pipeline
    final_state = research_graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nTopic: {final_state['topic']}")
    print(f"Sub-questions generated: {len(final_state['sub_questions'])}")
    print(f"Sources collected: {sum(len(r['sources']) for r in final_state['search_results'])}")
    print(f"Summaries written: {len(final_state['summaries'])}")
    print(f"\n--- FINAL REPORT ---\n")
    print(final_state["final_report"])