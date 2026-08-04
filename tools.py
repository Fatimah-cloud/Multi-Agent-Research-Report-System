from __future__ import annotations

import importlib.util
import os


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
    
    path = os.path.join(DRAFTS_DIR, f"{_slug(topic)}_rev{revision}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(draft)
    return path


def check_coverage(draft: str, subtopics: list[str]) -> dict:
  
    draft_lower = draft.lower()
    missing = [st for st in subtopics if st.lower() not in draft_lower]
    return {"missing": missing, "covered": [st for st in subtopics if st not in missing]}


def publish_report(topic: str, draft: str) -> str:
   
    path = os.path.join(PUBLISHED_DIR, f"{_slug(topic)}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(draft)
    return path


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:60]
