# Saathi — Offline Rural Health Assistant

> Fully offline triage & documentation aide for rural healthcare workers.
> Vision · Voice · Structured records — all on-device via Gemma 4 E4B + LiteRT-LM.

## Demo
- **Video:** https://youtu.be/Z1a5H0NCNEY
- **Hackathon:** Kaggle Build with Gemma — Kolkata 2026

## What it does

| Tab | Feature |
|-----|---------|
| 📷 Analyse Photo | Photograph a wound, rash, or prescription — Gemma describes it for the record |
| 📝 Auto-Fill Record | Describe a patient in plain language — Gemma extracts structured fields and saves to SQLite |
| 🎙️ Voice | Upload a voice note (≤30s) in English, Bengali, or Hindi — transcribed on-device |
| 📋 Records | Browse all saved patient records, colour-coded by severity |

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the model** (one-time, requires Hugging Face account)
```bash
# Accept the Gemma license at huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
huggingface-cli login
huggingface-cli download litert-community/gemma-4-E4B-it-litert-lm --local-dir ./models
```

**4. Run**
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`. The first load shows a "Loading Gemma 4 E4B on-device…" spinner — this is normal.

## How Gemma 4 is used

- **Vision:** Gemma 4's multimodal input analyses patient photos and documents. Image is passed before text — the order recommended in the model card.
- **Function calling:** LiteRT-LM auto-generates the JSON schema from Python type hints and docstrings. Gemma extracts structured fields and calls `fill_patient_record()` automatically.
- **ASR:** Voice notes transcribed on-device in English, Bengali, and Hindi. Audio is passed after the text prompt per the model card's ASR structure.

## Stack

- **Model:** Gemma 4 E4B via LiteRT-LM
- **UI:** Streamlit
- **Storage:** SQLite (local, no server)
- **Language:** Python

## Project structure

```
app.py        # Streamlit UI — 4 tabs
engine.py     # LiteRT-LM wrapper — singleton Engine, GPU→CPU fallback
tools.py      # SQLite function-calling tool + records viewer
requirements.txt
```

## Environment variable

Set `SAATHI_MODEL_PATH` to override the default model discovery:
```bash
set SAATHI_MODEL_PATH=C:\path\to\your\model.litertlm
```
