from __future__ import annotations

import importlib.util
import os
from typing import Annotated, TypedDict

from langgraph.graph import add_messages




class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    notes: Annotated[list[str], add_notes]

    topic: str
    subtopics: list[str]

    draft: str
    critique: str
    approved: bool
    revisions: int
    max_revisions: int
