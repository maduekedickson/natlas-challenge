import streamlit as st

import api_client

st.set_page_config(page_title="History - N-ATLaS", page_icon="📜", layout="wide")
st.title("📜 Report History")

try:
    facilities = ["(all)"] + api_client.list_facilities()
except Exception as e:
    st.error(f"Could not load facilities: {e}")
    facilities = ["(all)"]

facility_filter = st.selectbox("Filter by facility", facilities)
facility_param = None if facility_filter == "(all)" else facility_filter

try:
    reports = api_client.list_reports(limit=200, facility=facility_param)
except Exception as e:
    st.error(f"Could not load reports: {e}")
    reports = []

if not reports:
    st.info("No reports found.")
else:
    st.dataframe(reports, use_container_width=True)

    report_id = st.number_input("Open report by ID", min_value=1, step=1)
    if st.button("View details"):
        try:
            detail = api_client.get_report(int(report_id))
            st.subheader(f"Report #{detail['id']}")
            st.write("**Original text:**")
            st.write(detail["report_text"])
            st.write("**Structured data:**")
            st.json(detail["structured_data"])
            st.write("**Stock items:**")
            if detail["stock_items"]:
                st.table(detail["stock_items"])
            else:
                st.write("None")
        except Exception as e:
            st.error(f"Could not load report: {e}")
