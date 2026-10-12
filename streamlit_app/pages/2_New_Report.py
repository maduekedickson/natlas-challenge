import streamlit as st

import api_client

EXAMPLE_REPORTS = {
    "Vaccine + ORS shortage (English)": (
        "Good morning, this is Ngozi from Owerri Primary Health Centre. We have 20 "
        "doses of measles vaccine, we are out of ORS, and we offer antenatal care. "
        "The facility is open today."
    ),
    "Multiple shortages, Pidgin-flavoured": (
        "Abeg make I report, this is Musa from General Hospital, Kano. we are out of "
        "amoxicillin; we have 5 kits of malaria RDT kits; we are out of gloves. We "
        "offer malaria testing, general outpatient services here. The facility is "
        "closed today due to staff shortage."
    ),
    "Facility fully stocked": (
        "Daily report: This is Bisi from Ibadan Comprehensive Health Centre. we have "
        "150 doses of pentavalent vaccine; we have 80 tablets of paracetamol. We "
        "offer immunization, family planning, postnatal care here. The facility is "
        "open today."
    ),
}

st.set_page_config(page_title="New Report - N-ATLaS", page_icon="📝", layout="centered")
st.title("📝 New Report")
st.caption("Submit a free-text health facility report for extraction.")

with st.sidebar:
    st.subheader("Service status")
    try:
        st.success("API reachable")
        st.json(api_client.health())
    except Exception as e:
        st.error(f"Cannot reach API: {e}")

example_choice = st.selectbox("Example reports", ["(write your own)"] + list(EXAMPLE_REPORTS.keys()))
default_text = "" if example_choice == "(write your own)" else EXAMPLE_REPORTS[example_choice]
report_text = st.text_area("Report text", value=default_text, height=140)

if st.button("Analyze", type="primary", disabled=not report_text.strip()):
    with st.spinner("Running N-ATLaS inference..."):
        try:
            result = api_client.analyze(report_text)
        except Exception as e:
            st.error(f"Request failed: {e}")
            result = None

    if result:
        if result.get("report_id"):
            st.success(f"Saved as report #{result['report_id']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Facility", result.get("facility_name") or "—")
        col2.metric("Reporter", result.get("reporter_name") or "—")
        status = result.get("facility_status") or "—"
        col3.metric("Status", status.upper() if status != "—" else "—")

        st.markdown("**Stock items**")
        stock_items = result.get("stock_items") or []
        if stock_items:
            st.table(stock_items)
        else:
            st.write("None extracted.")

        st.markdown("**Services available**")
        services = result.get("services_available") or []
        st.write(", ".join(services) if services else "None extracted.")

        st.caption(
            f"Parsed OK: {result.get('parsed_ok')} · "
            f"Generation time: {result.get('generation_seconds')}s"
        )

        with st.expander("Raw model output"):
            st.code(result.get("raw_model_output", ""), language="json")
