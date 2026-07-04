"""Function-calling tools. LiteRT-LM auto-generates the schema from the
docstrings + type hints, so keep them accurate and descriptive."""

import sqlite3
import threading
from datetime import datetime

DB = "records.db"
VALID_SEVERITY = {"low", "medium", "high"}

# Per-thread storage so concurrent Streamlit sessions don't share state.
_ctx = threading.local()


def _init_db():
    con = sqlite3.connect(DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT, name TEXT, age INTEGER,
            symptoms TEXT, severity TEXT, notes TEXT
        )"""
    )
    con.commit()
    con.close()


def get_last_saved() -> dict:
    """Return the record saved during the most recent call on this thread."""
    return getattr(_ctx, "last_saved", {})


def get_all_records() -> list:
    """Return all patient records, most recent first."""
    _init_db()
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, created, name, age, symptoms, severity, notes "
        "FROM patients ORDER BY id DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def fill_patient_record(name: str, age: int, symptoms: str,
                        severity: str, notes: str = "") -> str:
    """Create a structured triage record for one patient and save it locally.

    Args:
        name: Patient's full name.
        age: Patient's age in years.
        symptoms: Observed or reported symptoms, in plain language.
        severity: Urgency level. Must be one of 'low', 'medium', 'high'.
        notes: Any extra context for the clinician (optional).
    """
    sev = severity.lower()
    if sev not in VALID_SEVERITY:
        return f"Invalid severity '{severity}'. Must be one of: low, medium, high."

    _init_db()
    row = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "name": name, "age": age, "symptoms": symptoms,
        "severity": sev, "notes": notes,
    }
    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO patients (created,name,age,symptoms,severity,notes) "
        "VALUES (:created,:name,:age,:symptoms,:severity,:notes)", row,
    )
    con.commit()
    con.close()
    _ctx.last_saved = row
    return f"Record saved for {name} (severity: {sev})."


TOOLS = [fill_patient_record]
