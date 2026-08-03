from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send

from state import TeamState
from tools import check_coverage, publish_report, save_draft, search_web, wiki_summary


class SubState(TypedDict):
    """Minimal input schema for a fanned-out sub_researcher instance.

    This is the map-reduce.ipynb pattern: each parallel Send carries just
    the one field the sub-task needs, not the whole TeamState.
    """
    subtopic: str


def _llm(temperature: float = 0):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"), temperature=temperature
    )


def _text(content) -> str:
    
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

def planner_node(state: TeamState, config, *, store: BaseStore) -> dict:
    topic = state["topic"]
    user_id = config["configurable"].get("user_id") or "default_user"
    namespace = ("research_memory", user_id)

    cached = store.get(namespace, topic)
    if cached is not None:
        # Long-term memory hit: reuse notes gathered in a PREVIOUS, separate
        # session instead of re-researching from scratch.
        return {
            "notes": cached.value["notes"],
            "subtopics": cached.value["subtopics"],
            "revisions": 0,
            "max_revisions": state.get("max_revisions", 2),
            "messages": [HumanMessage(content=f"[memory] Reusing prior research on '{topic}'.")],
        }

    # No memory hit -> split the topic into subtopics. A real system might
    # call the LLM for this; a light heuristic is enough to demonstrate the
    # routing and keeps the demo deterministic.
    parts = [p.strip() for p in topic.replace(" and ", ",").split(",") if p.strip()]
    subtopics = parts if len(parts) > 1 else [topic]

    return {
        "subtopics": subtopics,
        "revisions": 0,
        "max_revisions": state.get("max_revisions", 2),
    }


def route_after_planner(state: TeamState):
    """Conditional routing whose path depends on state (task requirement)."""
    if state.get("notes"):
        # memory-skip branch
        return "writer"
    if len(state["subtopics"]) > 1:
        # parallel branch (Send fan-out)
        return [Send("sub_researcher", {"subtopic": st}) for st in state["subtopics"]]
    # single, non-parallel branch
    return "single_research"


# --------------------------------------------------------------------------
# Researcher (parallel and single variants share this logic)
# --------------------------------------------------------------------------

def _research_one(subtopic: str) -> str:
    web = search_web.func(subtopic)
    wiki = wiki_summary.func(subtopic)
    return f"{subtopic} -- {web} | background: {wiki}"[:500]


def sub_researcher(state: SubState) -> dict:
    """Runs once per subtopic, in parallel, via Send. Output merges into the
    parent graph's `notes` field through the add_notes reducer."""
    return {"notes": [_research_one(state["subtopic"])]}


def single_research(state: TeamState) -> dict:
    """The non-parallel branch: one subtopic, called directly (no Send)."""
    return {"notes": [_research_one(state["subtopics"][0])]}


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------

def writer_node(state: TeamState) -> dict:
   
    prompt = (
        f"Write a tight 4-6 sentence report on '{state['topic']}', covering: "
        f"{', '.join(state['subtopics'])}.\n\nResearch notes:\n"
        + "\n".join(f"- {n}" for n in state["notes"])
    )
    if state.get("critique"):
        prompt += f"\n\nThe previous draft was reviewed and rejected for this reason: {state['critique']}\nRevise accordingly."

    draft = _text(_llm().invoke(prompt).content)
    save_draft(state["topic"], draft, state.get("revisions", 0))
    return {"draft": draft}


# --------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------

def critic_node(state: TeamState) -> dict:
    coverage = check_coverage(state["draft"], state["subtopics"])
    review_prompt = (
        f"Topic: {state['topic']}\nSubtopics that must be covered: {state['subtopics']}\n"
        f"Automated coverage check found these MISSING: {coverage['missing']}\n\n"
        f"Draft:\n{state['draft']}\n\n"
        "Reply with 'APPROVE' if the draft is accurate, complete, and covers "
        "every subtopic. Otherwise reply 'REVISE: <one sentence reason>'."
    )
    verdict = _text(_llm().invoke(review_prompt).content).strip()
    approved = verdict.upper().startswith("APPROVE") and not coverage["missing"]

    return {
        "approved": approved,
        "critique": verdict,
        "revisions": state.get("revisions", 0) + (0 if approved else 1),
    }


def route_after_critic(state: TeamState):
    if state["approved"]:
        return "publish"
    if state["revisions"] >= state.get("max_revisions", 2):
        # hard cap hit -> escalate/force-publish rather than loop forever
        return "publish"
    return "writer"  # loop-back


# --------------------------------------------------------------------------
# Publish (irreversible; sits behind the human breakpoint)
# --------------------------------------------------------------------------

def publish_node(state: TeamState, config, *, store: BaseStore) -> dict:
    path = publish_report(state["topic"], state["draft"])

    user_id = config["configurable"].get("user_id") or "default_user"
    namespace = ("research_memory", user_id)
    store.put(
        namespace,
        state["topic"],
        {"subtopics": state["subtopics"], "notes": state["notes"], "draft": state["draft"]},
    )

    return {"messages": [HumanMessage(content=f"[system] Published to {path}")]}


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(TeamState)
    graph.add_node("planner", planner_node)
    graph.add_node("sub_researcher", sub_researcher)
    graph.add_node("single_research", single_research)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("publish", publish_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"writer": "writer", "single_research": "single_research", "sub_researcher": "sub_researcher"},
    )
    graph.add_edge("sub_researcher", "writer")
    graph.add_edge("single_research", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"writer": "writer", "publish": "publish"})
    graph.add_edge("publish", END)

    return graph


# Compiled graph for `langgraph dev` / LangGraph Studio to import directly.
# No explicit checkpointer or store here -- the Studio dev server provides
# its own persistence layer and injects it into any node that asks for a
# `store`/`config` parameter, same as planner_node and publish_node do here.
# The human breakpoint stays active so you can watch it pause live in Studio.
graph = build_graph().compile(interrupt_before=["publish"])
