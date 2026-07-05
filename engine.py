"""LiteRT-LM model wrapper — loads Gemma 4 E4B once and vends conversations."""

import glob
import os
import contextlib
import litert_lm

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _find_model_path() -> str:
    """Return the model path, checking env var first then ./models/."""
    explicit = os.environ.get("SAATHI_MODEL_PATH")
    if explicit:
        return explicit
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        glob.glob(os.path.join(base, "models", "*.task")) or
        glob.glob(os.path.join(base, "models", "*.litertlm"))
    )
    if not candidates:
        raise FileNotFoundError(
            "No model file found in ./models/.\n"
            "Download it with:\n"
            "  huggingface-cli download litert-community/gemma-4-E4B-it-litert-lm "
            "--local-dir ./models"
        )
    return candidates[0]


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
