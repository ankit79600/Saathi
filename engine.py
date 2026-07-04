"""LiteRT-LM model wrapper — loads Gemma 4 E4B once and vends conversations."""

import glob
import os
import contextlib
import litert_lm

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _find_model_path() -> str:
    """Return the .task model path, checking env var first then ./models/."""
    explicit = os.environ.get("SAATHI_MODEL_PATH")
    if explicit:
        return explicit
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(os.path.join(base, "models", "*.task"))
    if not candidates:
        raise FileNotFoundError(
            "No .task model file found in ./models/.\n"
            "Download it with:\n"
            "  huggingface-cli download litert-community/gemma-4-E4B-it-litert-lm "
            "--local-dir ./models"
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Singleton model — loaded once per process, reused across Streamlit reruns
# ---------------------------------------------------------------------------

_model = None


def load_model() -> litert_lm.GenAiModel:
    """Load the model singleton, trying GPU first then falling back to CPU."""
    global _model
    if _model is None:
        path = _find_model_path()
        for backend in ("gpu", "cpu"):
            try:
                _model = litert_lm.GenAiModel(model_path=path, backend=backend)
                break
            except Exception:
                if backend == "cpu":
                    raise
    return _model


# ---------------------------------------------------------------------------
# Conversation wrapper
# ---------------------------------------------------------------------------

class _Conversation:
    """Wraps a litert_lm session with a stable send_message interface."""

    def __init__(self, session) -> None:
        self._session = session

    def send_message(self, content) -> dict:
        """Send a message and return {"content": [{"text": "..."}]}."""
        raw = self._session.send_message(content)

        # Normalise whatever litert_lm returns into the shape app.py expects.
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            text = raw
        elif hasattr(raw, "text"):
            text = raw.text
        else:
            text = str(raw)

        return {"content": [{"text": text}]}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def new_conversation(tools: list | None = None):
    """Context manager that yields a Gemma 4 E4B conversation session.

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
    session = model.create_session(**kwargs)
    try:
        yield _Conversation(session)
    finally:
        session.close()
