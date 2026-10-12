import streamlit as st

import api_client

st.set_page_config(page_title="Dashboard - N-ATLaS", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

try:
    stats = api_client.dashboard_stats()
except Exception as e:
    st.error(f"Could not load dashboard: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Total reports", stats["total_reports"])
col2.metric("Facilities represented", stats["facilities_represented"])
col3.metric("Reports today", stats["reports_today"])

col4, col5, col6 = st.columns(3)
col4.metric("Open-status reports", stats["open_facilities"])
col5.metric("Closed-status reports", stats["closed_facilities"])
col6.metric("Stock-out observations", stats["stock_out_observations"])

st.subheader("Recent reports")
recent = stats.get("recent_reports", [])
if recent:
    st.dataframe(recent, use_container_width=True)
else:
    st.info("No reports submitted yet. Go to 'New Report' to submit one.")
