"""
Task 2 tools. `search_web` and `wiki_summary` are imported straight from
Task 1 -- the Researcher role reuses them unchanged. The Writer, Critic and
Publish roles get their own tools, because they do genuinely different jobs.
"""

from __future__ import annotations

import importlib.util
import os

# Task 1 has its own module literally named `tools.py`; load it under a
# distinct module name so this file (task2's own tools.py) doesn't collide
# with it on `sys.modules["tools"]`.
_task1_tools_path = os.path.join(os.path.dirname(__file__), "..", "task1", "tools.py")
_spec = importlib.util.spec_from_file_location("task1_tools", _task1_tools_path)
_task1_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task1_tools)
search_web = _task1_tools.search_web  # reused from Task 1
wiki_summary = _task1_tools.wiki_summary  # reused from Task 1

DRAFTS_DIR = os.path.join(os.path.dirname(__file__), "drafts")
PUBLISHED_DIR = os.path.join(os.path.dirname(__file__), "published")
os.makedirs(DRAFTS_DIR, exist_ok=True)
os.makedirs(PUBLISHED_DIR, exist_ok=True)


def save_draft(topic: str, draft: str, revision: int) -> str:
    """Write the current draft to disk (Writer role). Reversible - just a scratch file."""
    path = os.path.join(DRAFTS_DIR, f"{_slug(topic)}_rev{revision}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(draft)
    return path


def check_coverage(draft: str, subtopics: list[str]) -> dict:
    """Deterministic check (Critic role): which subtopics does the draft actually mention?

    This is intentionally NOT an LLM call -- a cheap, reliable check the
    critic uses alongside its own judgment, similar to a linter feeding a
    code reviewer.
    """
    draft_lower = draft.lower()
    missing = [st for st in subtopics if st.lower() not in draft_lower]
    return {"missing": missing, "covered": [st for st in subtopics if st not in missing]}


def publish_report(topic: str, draft: str) -> str:
    """Publish the final report (irreversible action -- this is what the
    human-in-the-loop breakpoint sits in front of).
    """
    path = os.path.join(PUBLISHED_DIR, f"{_slug(topic)}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(draft)
    return path


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:60]
