"""Generic speech-to-text transcription.

NOT N-ATLaS: the NCAIR1/N-ATLaS repository contains only the LLM, no bundled
ASR component. This uses openai-whisper, a separate, clearly-labelled ASR
step. Its output must be shown to the user for editing before being sent to
N-ATLaS for extraction -- see streamlit_app/pages/3_Voice_Report.py.
"""

from __future__ import annotations

import logging

from api.config import WHISPER_MODEL

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        import whisper

        logger.info("Loading Whisper model (%s) for speech-to-text", WHISPER_MODEL)
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe(audio_path: str) -> str:
    model = _get_model()
    result = model.transcribe(audio_path)
    return result["text"].strip()
