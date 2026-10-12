import streamlit as st

import api_client

st.set_page_config(page_title="Exports - N-ATLaS", page_icon="⬇️", layout="centered")
st.title("⬇️ Exports")

try:
    facilities = ["(all)"] + api_client.list_facilities()
except Exception as e:
    st.error(f"Could not load facilities: {e}")
    facilities = ["(all)"]

facility_filter = st.selectbox("Filter by facility (optional)", facilities)
facility_param = None if facility_filter == "(all)" else facility_filter

st.subheader("Report-level export")
st.caption("One row per report, with the full structured JSON kept as a column.")
col1, col2 = st.columns(2)
with col1:
    st.link_button("Download CSV", api_client.export_url("reports", "csv", facility_param))
with col2:
    st.link_button("Download Excel", api_client.export_url("reports", "xlsx", facility_param))

st.subheader("Stock-item-level export")
st.caption("One row per report x stock item — better suited to spreadsheet pivoting.")
col3, col4 = st.columns(2)
with col3:
    st.link_button("Download CSV", api_client.export_url("stock-items", "csv", facility_param))
with col4:
    st.link_button("Download Excel", api_client.export_url("stock-items", "xlsx", facility_param))
