# Multi-Agent Research & Report System

A 4-agent LangGraph system that researches a topic, drafts a
report, critiques it, and publishes it behind a human approval step.



## The agents

| Role | Job | Tools |
|---|---|---|
| **Planner** | splits the topic into subtopics; checks long-term memory for a prior run on this topic | — |
| **Researcher** | gathers info per subtopic — in parallel (`Send` fan-out) if there are multiple subtopics, or as a single call if there's one | `search_web`, `wiki_summary` |
| **Writer** | drafts a report from the research notes | `save_draft` |
| **Critic** | checks the draft covers every subtopic and reviews it; can send it back for revision (capped) | `check_coverage` |

Writer and Critic are separate agents on purpose — one agent reviewing its
own work has no real incentive to catch its own mistakes.

## What it demonstrates

- **Conditional routing** — the path depends on state: parallel fan-out,
  a single direct call, or reusing cached research from a past session
- **Parallel step + reducer** — `Send` fans out one `sub_researcher` per
  subtopic; results merge into `notes` 
- **Loop-back with a cap** — rejected drafts return to the Writer, capped
  by `max_revisions`
- **Human-in-the-loop** — execution pauses before `publish` (the one
  irreversible step) for approval
- **Long-term memory** — an `InMemoryStore` lets a brand-new session reuse
  research from a previous one
- **Time travel** — `get_state_history` locates the exact checkpoint where
  a draft was rejected, to debug why

## Output

- `graph.png` — the rendered graph diagram
- `drafts/` — every draft, including rejected ones
- `published/` — final approved reports
