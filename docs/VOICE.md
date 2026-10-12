# Voice Input

## What this is — and isn't

Voice reports are transcribed using **`openai-whisper`** (the `base` model by default, configurable via `WHISPER_MODEL`). This is a generic, well-supported, reproducible speech-to-text library — **not N-ATLaS's own ASR.**

The `NCAIR1/N-ATLaS` Hugging Face repository (inspected directly as part of this project — see `docs/FEASIBILITY.md`) contains only the LLM; there is no bundled ASR component to reuse. Whisper was chosen because it's a standard, modest dependency that runs acceptably on CPU or GPU, rather than introducing something exotic for a feature outside the project's core evaluated scope.

**The UI never claims this is N-ATLaS transcribing audio.** `streamlit_app/pages/3_Voice_Report.py` labels it plainly as "speech-to-text."

## Flow

```
Record/upload audio
        │
        ▼
POST /api/v1/transcribe  (Whisper)
        │
        ▼
Transcript shown in an EDITABLE text box
        │
        ▼ (only on explicit user action)
POST /api/v1/analyze     (the same, unchanged N-ATLaS pipeline)
```

**The transcript is never sent directly to N-ATLaS without being shown to the user first.** This is enforced by the UI flow (two separate buttons: "Transcribe" then "Analyze transcript"), not just a suggestion — there's no code path that skips the review step.

## Language scope

English only, matching the rest of this project's dataset/benchmark scope (see `docs/DATASET.md`'s own documented limitation on native Igbo/Hausa/Yoruba support). Whisper itself is multilingual, but N-ATLaS's extraction accuracy on non-English transcripts hasn't been evaluated here — out of scope for this pass.

## Requirements

- `openai-whisper` (Python package, in `requirements.txt`)
- **`ffmpeg`** (system binary, used by Whisper to decode audio) — present by default on Colab's standard image. If running elsewhere and transcription fails with an ffmpeg-related error, install it (`apt-get install -y ffmpeg` on Debian/Ubuntu).
- Recording support (`st.audio_input`) requires Streamlit ≥ 1.40; on older versions the page falls back to file upload only (guarded with `hasattr(st, "audio_input")`, not a hard failure).

## Known limitation

Not evaluated against the 520-example benchmark — the benchmark is text-only extraction, deliberately untouched by this feature (see `docs/IMPLEMENTATION_PLAN.md` §21, benchmark integrity). Voice-path quality is therefore a combination of unverified Whisper transcription accuracy plus the already-measured text-extraction accuracy, not a new independently-measured number.
