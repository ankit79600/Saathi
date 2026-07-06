"""LiteRT-LM model wrapper — loads Gemma 4 E4B once and vends conversations."""

import glob
import os
import contextlib
import litert_lm

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_HF_REPO = "litert-community/gemma-4-E4B-it-litert-lm"
# Exact filename of the Linux/Android native model — the repo also ships a
# *-web.litertlm (WASM/browser) and a *-web.task that both fail on Linux litert-lm.
_MODEL_FILENAME = "gemma-4-E4B-it.litertlm"


def _find_model_path() -> str:
    """Return the model path: env var → ./models/ → auto-download from HF Hub."""
    explicit = os.environ.get("SAATHI_MODEL_PATH")
    if explicit:
        return explicit

    base = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(base, "models")
    os.makedirs(local_dir, exist_ok=True)

    # Only accept the native device model — skip *-web.* variants (WASM/browser only)
    candidates = sorted(
        f for f in glob.glob(os.path.join(local_dir, "*.litertlm"))
        if "-web" not in os.path.basename(f)
    )
    if candidates:
        return candidates[0]

    # Download only the specific native model file — avoids pulling web variants.
    # Requires HF_TOKEN env var (Gemma is gated; accept license on huggingface.co first).
    print(f"Model not found in {local_dir}. Downloading {_MODEL_FILENAME} from {_HF_REPO} …")
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=_MODEL_FILENAME,
        local_dir=local_dir,
    )
    return model_path


# ---------------------------------------------------------------------------
# Singleton model — loaded once per process, reused across Streamlit reruns
# ---------------------------------------------------------------------------

_model = None


def load_model() -> litert_lm.Engine:
    """Load the Engine singleton, trying GPU first then falling back to CPU."""
    global _model
    if _model is None:
        path = _find_model_path()
        for backend in (litert_lm.Backend.GPU(), litert_lm.Backend.CPU()):
            try:
                _model = litert_lm.Engine(
                    model_path=path,
                    backend=backend,
                    vision_backend=backend,
                    audio_backend=backend,
                )
                break
            except Exception:
                if isinstance(backend, litert_lm.Backend.CPU):
                    raise
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def new_conversation(tools: list | None = None):
    """Context manager that yields a Gemma 4 E4B conversation.

    Args:
        tools: Optional list of Python callables for function calling.
               LiteRT-LM derives the JSON schema from their type hints and
               docstrings automatically.

    Usage::

        with new_conversation() as convo:
            resp = convo.send_message("Hello")
            print(resp["content"][0]["text"])
    """
    model = load_model()
    kwargs = {"tools": tools} if tools else {}
    convo = model.create_conversation(**kwargs)
    yield convo
