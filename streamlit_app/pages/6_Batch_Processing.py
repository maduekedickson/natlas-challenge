import streamlit as st

import api_client

st.set_page_config(page_title="Batch Processing - N-ATLaS", page_icon="📦", layout="wide")
st.title("📦 Batch Processing")
st.caption(
    "Upload a CSV with a 'report' column (optionally 'facility' and 'reporter' "
    "columns too). Each row is processed through the same N-ATLaS pipeline as "
    "a single report — no separate inference path."
)

st.code(
    'facility,reporter,report\n'
    'Asaba PHC,John,"We have 20 doses of measles vaccine, open today."\n'
    'Ika PHC,Mary,"We are out of ORS, facility closed."',
    language="text",
)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded and st.button("Process batch", type="primary"):
    with st.spinner("Processing batch — this runs real inference per row, may take a while..."):
        try:
            result = api_client.batch_process(uploaded.getvalue(), uploaded.name)
        except Exception as e:
            st.error(f"Batch processing failed: {e}")
            result = None

    if result:
        if "error" in result:
            st.error(result["error"])
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total rows", result["total_rows"])
            col2.metric("Successful", result["successful"])
            col3.metric("Failed", result["failed"])
            col4.metric("Time", f"{result['processing_seconds']}s")

            if result["failures"]:
                st.subheader("Failures")
                st.dataframe(result["failures"], use_container_width=True)

            if result["successes"]:
                st.subheader("Successful rows")
                st.dataframe(result["successes"], use_container_width=True)
                st.info("Processed reports are saved — see History, Trends, and Exports for them.")
