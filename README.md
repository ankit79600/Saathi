# Saathi — Offline Rural Health Assistant (Gemma 4 E4B)

Fully offline triage & documentation aide for rural health workers.
Vision + voice + structured form-filling, all on-device via LiteRT-LM.

## Setup
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt huggingface_hub`
3. Accept the Gemma license on Hugging Face (in your browser), then:
   `huggingface-cli login`
   `huggingface-cli download litert-community/gemma-4-E4B-it-litert-lm --local-dir ./models`
4. Confirm the model filename in ./models matches MODEL_PATH in engine.py
5. `streamlit run app.py`

## How Gemma 4 is used
- **Vision:** analyses patient photos & documents (OCR) — image-before-text order.
- **Function calling:** auto-fills structured patient records from free text.
- **Audio:** on-device ASR for voice notes (≤30s).
- **Edge/offline:** GPU backend + speculative decoding (MTP); no internet after download.