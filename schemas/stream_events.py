from typing import Any, Literal
from pydantic import BaseModel, Field

class StreamEvent(BaseModel):
    version: Literal[1] = 1
    type: Literal["token", "activity", "interrupt", "error", "done"]
    session_id: str
    data: dict[str, Any] = Field(default_factory=dict)
