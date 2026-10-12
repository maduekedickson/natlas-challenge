"""Landing page for the N-ATLaS Health Intelligence Platform.

Actual features live in streamlit_app/pages/ (Streamlit's native multipage
mechanism -- each file there becomes a sidebar entry automatically).
"""

import streamlit as st

import api_client

st.set_page_config(page_title="N-ATLaS Health Intelligence", page_icon="🏥", layout="wide")

st.title("🏥 N-ATLaS Health Intelligence Platform")
st.caption(
    "N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation "
    "and Digital Economy, powered by Awarri Technologies."
)

st.info(
    "**This is an information-extraction and operational reporting tool. "
    "It is not a diagnostic system and does not provide medical advice, "
    "treatment recommendations, or clinical decision-making.**"
)

st.markdown(
    """
This platform extracts structured information — stock levels, services offered,
facility open/closed status — from free-text health facility reports, using a
LoRA-fine-tuned N-ATLaS.

Use the sidebar to navigate:

- **Dashboard** — operational statistics across all submitted reports
- **New Report** — submit a text report for analysis
- **Voice Report** — submit a report by voice (transcribed, then reviewed before analysis)
- **History** — browse and inspect past reports
- **Trends** — compare stock levels and facility status over time
- **Batch Processing** — upload a CSV of multiple reports at once
- **Exports** — download report data as CSV/Excel
"""
)

with st.sidebar:
    st.subheader("Service status")
    try:
        h = api_client.health()
        st.success("API reachable")
        st.json(h)
    except Exception as e:
        st.error(f"Cannot reach the API backend: {e}")
