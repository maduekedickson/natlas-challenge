import pandas as pd
import streamlit as st

import api_client

st.set_page_config(page_title="Trends - N-ATLaS", page_icon="📈", layout="wide")
st.title("📈 Trends & Comparison")

try:
    facilities = api_client.list_facilities()
    items = api_client.list_stock_items()
except Exception as e:
    st.error(f"Could not load filter options: {e}")
    facilities, items = [], []

if not facilities:
    st.info("No data yet — submit some reports first (New Report, Voice Report, or Batch Processing).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Stock quantity over time", "Facility status over time", "Overview"])

with tab1:
    facility = st.selectbox("Facility", facilities, key="trend_facility")
    item = st.selectbox("Commodity", items, key="trend_item") if items else None
    if item and st.button("Show trend"):
        data = api_client.stock_trend(facility, item)
        if not data:
            st.warning("No data for this facility/commodity combination yet.")
        else:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            st.line_chart(df.set_index("date")["quantity"])
            st.dataframe(df, use_container_width=True)

with tab2:
    facility2 = st.selectbox("Facility", facilities, key="status_facility")
    if st.button("Show status history"):
        data = api_client.status_trend(facility2)
        if not data:
            st.warning("No data for this facility yet.")
        else:
            st.dataframe(pd.DataFrame(data), use_container_width=True)

with tab3:
    st.subheader("Reports by facility")
    rbf = api_client.reports_by_facility()
    if rbf:
        st.bar_chart(pd.DataFrame(rbf).set_index("facility")["count"])
    else:
        st.info("No data yet.")

    st.subheader("Commodity frequency")
    cf = api_client.commodity_frequency()
    if cf:
        st.bar_chart(pd.DataFrame(cf).set_index("item")["count"])
    else:
        st.info("No data yet.")

    st.subheader("Stock-out occurrences")
    so = api_client.stockouts()
    if so:
        st.bar_chart(pd.DataFrame(so).set_index("item")["count"])
    else:
        st.info("No stock-outs recorded yet.")
