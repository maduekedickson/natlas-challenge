import streamlit as st

import api_client

st.set_page_config(page_title="Exports - N-ATLaS", page_icon="⬇️", layout="centered")
st.title("⬇️ Exports")

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

try:
    facilities = ["(all)"] + api_client.list_facilities()
except Exception as e:
    st.error(f"Could not load facilities: {e}")
    facilities = ["(all)"]

facility_filter = st.selectbox("Filter by facility (optional)", facilities)
facility_param = None if facility_filter == "(all)" else facility_filter


def _download(label: str, kind: str, fmt: str, file_name: str, mime: str) -> None:
    try:
        data = api_client.export_bytes(kind, fmt, facility_param)
        st.download_button(label, data=data, file_name=file_name, mime=mime)
    except Exception as e:
        st.error(f"{label} failed: {e}")


st.subheader("Report-level export")
st.caption("One row per report, with the full structured JSON kept as a column.")
col1, col2 = st.columns(2)
with col1:
    _download("Download CSV", "reports", "csv", "reports.csv", "text/csv")
with col2:
    _download("Download Excel", "reports", "xlsx", "reports.xlsx", XLSX_MIME)

st.subheader("Stock-item-level export")
st.caption("One row per report x stock item — better suited to spreadsheet pivoting.")
col3, col4 = st.columns(2)
with col3:
    _download("Download CSV", "stock-items", "csv", "stock_items.csv", "text/csv")
with col4:
    _download("Download Excel", "stock-items", "xlsx", "stock_items.xlsx", XLSX_MIME)
