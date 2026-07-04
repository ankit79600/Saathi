"""Saathi — offline rural-health triage & documentation assistant.
Runs entirely on-device on Gemma 4 E4B via LiteRT-LM.

Run with:  streamlit run app.py
"""

import tempfile
import streamlit as st
import litert_lm

import engine
from tools import TOOLS, get_last_saved, get_all_records

st.set_page_config(page_title="Saathi · Offline Health Assistant", page_icon="🩺")
st.title("🩺 Saathi — Offline Rural Health Assistant")
st.caption("Powered by Gemma 4 E4B · runs fully offline on your GPU · "
           "a documentation & triage aide, not a diagnostic tool")


@st.cache_resource(show_spinner="Loading Gemma 4 E4B on-device…")
def _warmup():
    engine.load_model()
    return True


_warmup()

tab_vision, tab_form, tab_voice, tab_records = st.tabs(
    ["📷 Analyse Photo", "📝 Auto-Fill Record", "🎙️ Voice (optional)", "📋 Records"]
)

# ---------- 1. VISION: analyse a patient photo or document ----------
with tab_vision:
    st.subheader("Capture or upload an image")
    img = st.camera_input("Take a photo") or st.file_uploader(
        "…or upload", type=["jpg", "jpeg", "png"]
    )
    detail = st.radio("Detail level", ["Fast (captioning)", "High (OCR/documents)"],
                      horizontal=True)
    if img and st.button("Analyse with Gemma"):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img.getvalue())
            path = f.name
        prompt = (
            "Objectively describe what is visible in this image for a health "
            "worker's record. If it is a document or prescription, transcribe "
            "the key text. Do NOT diagnose. Note anything that looks urgent."
        )
        try:
            with st.spinner("Running on-device inference…"):
                with engine.new_conversation() as convo:
                    # Image BEFORE text = recommended modality order for Gemma 4.
                    resp = convo.send_message(litert_lm.Contents.of(
                        litert_lm.Content.ImageFile(absolute_path=path),
                        prompt,
                    ))
            text = resp["content"][0]["text"]
            st.session_state["vision_text"] = text
            st.success("Analysis complete")
            st.write(text)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

# ---------- 2. FUNCTION CALLING: turn notes into a structured record ----------
with tab_form:
    st.subheader("Describe the patient — Gemma fills the record for you")
    default = st.session_state.get("vision_text", "")
    desc = st.text_area("Case description (voice/photo output can seed this)",
                        value=default, height=140,
                        placeholder="e.g. Asha, 34, fever and rash for 3 days, "
                                    "seems moderate.")
    if st.button("Create structured record"):
        try:
            with st.spinner("Gemma is extracting fields and saving…"):
                with engine.new_conversation(tools=TOOLS) as convo:
                    reply = convo.send_message(
                        "From this description, call fill_patient_record with the "
                        "correct fields. Description: " + desc
                    )
            st.write(reply["content"][0]["text"])
            saved = get_last_saved()
            if saved:
                st.success("Saved locally ✅")
                st.json(saved)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

# ---------- 3. AUDIO (optional bonus): voice note -> text ----------
with tab_voice:
    st.subheader("Transcribe a short voice note (≤ 30s)")
    st.info("Optional feature. In a noisy venue, prefer the text/photo tabs.")
    audio = st.file_uploader("Upload a .wav clip", type=["wav"])
    lang = st.selectbox("Spoken language", ["English", "Bengali", "Hindi"])
    if audio and st.button("Transcribe"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio.getvalue())
            apath = f.name
        # ASR prompt structure from the model card; audio goes AFTER the text.
        asr = (f"Transcribe the following speech segment in {lang} into {lang} "
               "text. Only output the transcription, with no newlines.")
        try:
            with st.spinner("Transcribing on-device…"):
                with engine.new_conversation() as convo:
                    resp = convo.send_message(litert_lm.Contents.of(
                        asr,
                        litert_lm.Content.AudioFile(absolute_path=apath),
                    ))
            text = resp["content"][0]["text"]
            st.session_state["vision_text"] = text
            st.success("Transcribed — now open the Auto-Fill Record tab")
            st.write(text)
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")

# ---------- 4. RECORDS: browse saved patient records ----------
with tab_records:
    st.subheader("Saved patient records")
    if st.button("Refresh"):
        st.rerun()
    records = get_all_records()
    if not records:
        st.info("No records yet. Use the Auto-Fill Record tab to create one.")
    else:
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for r in records:
            icon = severity_icon.get(r["severity"], "⚪")
            with st.expander(f"{icon} {r['name']}, {r['age']} — {r['created']}"):
                st.write(f"**Symptoms:** {r['symptoms']}")
                st.write(f"**Severity:** {r['severity']}")
                if r["notes"]:
                    st.write(f"**Notes:** {r['notes']}")
        st.caption(f"{len(records)} record(s) stored in records.db")
