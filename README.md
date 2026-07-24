# Multi-Agent Research Assistant

Give it any topic. Four AI agents collaborate to research it and produce a structured report — automatically.

## How It Works

A user types a research topic. Then:

1. **Planner** breaks it into 4 specific sub-questions
2. **Researcher** searches the web for each sub-question (via Tavily)
3. **Summarizer** turns raw search results into clean summaries
4. **Writer** combines everything into a final structured report

All 4 agents share a single state object managed by LangGraph — no agent calls another directly, they just read and write to shared state.

## Tech Stack

- LangGraph — agent orchestration and state management
- Google Gemini API — powers Planner, Summarizer, Writer agents
- Tavily Search API — real-time web search for the Researcher agent
- Streamlit — dashboard with custom theming

## Features

- Real-time web search (not pre-loaded data)
- Modular agent design — each agent has one job
- Research history tracked across sessions
- Stats per run: sources found, word count, time taken
- Download final report as .txt

## API Keys Needed

- Google Gemini API — https://aistudio.google.com/apikey
- Tavily Search API — https://tavily.com
