"""
Task 2 state schema. This literally extends Task 1's schema rather than
reinventing it: `messages` (add_messages) and `notes` (add_notes, the
dedup-append reducer from Task 1) are imported straight from task1/state.py.
We add the fields a 4-agent team needs on top of that.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Annotated, TypedDict

from langgraph.graph import add_messages

# Task 1 also has a module literally named `state.py`, so a plain
# `sys.path` + `import state` would collide with this file. Load it under
# a distinct module name instead, to make the "reused from Task 1" import
# explicit and unambiguous.
_task1_state_path = os.path.join(os.path.dirname(__file__), "..", "task1", "state.py")
_spec = importlib.util.spec_from_file_location("task1_state", _task1_state_path)
_task1_state = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task1_state)
add_notes = _task1_state.add_notes  # reused from Task 1 on purpose


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
