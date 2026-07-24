import streamlit as st
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.research_graph import build_research_graph

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for visual differentiation
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* Agent cards in sidebar */
    .agent-card {
        background: #1A1A2E;
        border-left: 3px solid #6C63FF;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    .agent-card h4 {
        color: #6C63FF;
        margin: 0 0 0.2rem 0;
        font-size: 0.85rem;
    }
    .agent-card p {
        color: #aaa;
        margin: 0;
        font-size: 0.78rem;
    }

    /* Metric cards */
    .metric-card {
        background: #1A1A2E;
        border: 1px solid #2A2A4A;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 0.2rem;
    }

    /* Report box */
    .report-box {
        background: #1A1A2E;
        border: 1px solid #2A2A4A;
        border-radius: 10px;
        padding: 1.5rem;
        line-height: 1.8;
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <span style='font-size:2rem'>🔬</span>
        <h2 style='color:#6C63FF; margin:0.3rem 0'>Agent Pipeline</h2>
        <p style='color:#aaa; font-size:0.8rem'>4 AI agents collaborating</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="agent-card">
        <h4> Agent 1 — Planner</h4>
        <p>Breaks topic into focused sub-questions</p>
    </div>
    <div class="agent-card">
        <h4> Agent 2 — Researcher</h4>
        <p>Searches the web via Tavily API</p>
    </div>
    <div class="agent-card">
        <h4> Agent 3 — Summarizer</h4>
        <p>Condenses raw results into summaries</p>
    </div>
    <div class="agent-card">
        <h4> Agent 4 — Writer</h4>
        <p>Writes the final structured report</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Built with LangGraph · Gemini API · Tavily Search")

    # Research history in sidebar
    st.subheader(" Recent Topics")
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            st.caption(f"• {h}")
    else:
        st.caption("No research yet")

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔬 Multi-Agent Research Assistant</h1>
    <p>Powered by 4 specialized AI agents: Planner → Researcher → Summarizer → Writer</p>
</div>
""", unsafe_allow_html=True)

# Topic input
col1, col2 = st.columns([5, 1])
with col1:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Future of Electric Vehicles in India",
        label_visibility="collapsed"
    )
with col2:
    run_button = st.button(" Research", type="primary", use_container_width=True)

if run_button:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        # Add to history
        if topic not in st.session_state.history:
            st.session_state.history.append(topic)

        with st.status("🔄 Running agent pipeline...", expanded=True) as status:
            st.write(" Agent 1 (Planner) — Generating sub-questions...")
            start_time = time.time()

            research_graph = build_research_graph()
            initial_state = {
                "topic": topic,
                "sub_questions": [],
                "search_results": [],
                "summaries": [],
                "final_report": ""
            }

            final_state = research_graph.invoke(initial_state)
            elapsed = round(time.time() - start_time, 1)
            status.update(label=f" Research complete in {elapsed}s!", state="complete")

        # Metrics row
        st.subheader(" Research Stats")
        total_sources = sum(len(r["sources"]) for r in final_state["search_results"])
        word_count = len(final_state["final_report"].split())
        report_chars = len(final_state["final_report"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(final_state['sub_questions'])}</div>
                <div class="metric-label">Sub-questions</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_sources}</div>
                <div class="metric-label">Sources found</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{word_count}</div>
                <div class="metric-label">Words in report</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{elapsed}s</div>
                <div class="metric-label">Time taken</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Agent Activity Log
        st.subheader(" Agent Activity Log")

        with st.expander(" Agent 1 — Planner Output", expanded=True):
            for i, q in enumerate(final_state["sub_questions"], 1):
                st.markdown(f"**{i}.** {q}")

        with st.expander(" Agent 2 — Researcher Output", expanded=False):
            for item in final_state["search_results"]:
                st.markdown(f"**Q:** {item['question']}")
                for s in item["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")
                st.divider()

        with st.expander(" Agent 3 — Summarizer Output", expanded=False):
            for item in final_state["summaries"]:
                st.markdown(f"**Q:** {item['question']}")
                st.info(item["summary"])
                st.divider()

        st.divider()

        # Final report
        st.subheader(" Final Research Report")
        st.markdown(
            f'<div class="report-box">{final_state["final_report"]}</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # Download
        st.download_button(
            label=" Download Report (.txt)",
            data=final_state["final_report"],
            file_name=f"research_{topic[:30].replace(' ', '_')}.txt",
            mime="text/plain",
            key="download_report"
        )


        # Footer
        st.markdown("""
        <hr style='border: 1px solid #2A2A4A; margin-top: 3rem;'>
        <div style='text-align: center; padding: 1rem 0; color: #555; font-size: 0.8rem;'>
            Built by <span style='color: #6C63FF; font-weight: 600;'>Tulaja Patil</span> 
            &nbsp;·&nbsp; 
            Powered by LangGraph · Gemini · Tavily
        </div>
        """, unsafe_allow_html=True)

