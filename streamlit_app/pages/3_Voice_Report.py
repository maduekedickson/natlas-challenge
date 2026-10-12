import streamlit as st

import api_client

st.set_page_config(page_title="Voice Report - N-ATLaS", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Report")
st.caption(
    "Generic speech-to-text transcription (not N-ATLaS's own ASR). "
    "Review and edit the transcript below before it is analyzed."
)

audio = st.audio_input("Record your report") if hasattr(st, "audio_input") else None
uploaded = st.file_uploader("...or upload an audio file", type=["wav", "mp3", "m4a", "ogg"])

audio_bytes = None
filename = "audio.wav"
if audio is not None:
    audio_bytes = audio.getvalue()
elif uploaded is not None:
    audio_bytes = uploaded.getvalue()
    filename = uploaded.name

if audio_bytes and st.button("Transcribe"):
    with st.spinner("Transcribing..."):
        try:
            transcript = api_client.transcribe(audio_bytes, filename)
            st.session_state["voice_transcript"] = transcript
        except Exception as e:
            st.error(f"Transcription failed: {e}")

transcript = st.text_area(
    "Transcript (edit before analyzing)",
    value=st.session_state.get("voice_transcript", ""),
    height=140,
)

if st.button("Analyze transcript", type="primary", disabled=not transcript.strip()):
    with st.spinner("Running N-ATLaS inference..."):
        try:
            result = api_client.analyze(transcript)
        except Exception as e:
            st.error(f"Request failed: {e}")
            result = None
    if result:
        if result.get("report_id"):
            st.success(f"Saved as report #{result['report_id']}")
        st.json(result)
