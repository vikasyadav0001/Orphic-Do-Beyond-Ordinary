# utils/interrupts.py
from typing import Any

def read_resume_value(value: Any) -> tuple[str, dict]:
    """
    Safely normalizes the resume value returned from LangGraph's interrupt().
    Returns (decision, input_dict)
    """
    if isinstance(value, str):
        return value, {}

    if isinstance(value, dict):
        return value.get("decision", "cancel"), value.get("input", {})

    return "cancel", {}
