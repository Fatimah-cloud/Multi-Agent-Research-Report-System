from __future__ import annotations

import os
os.environ["GOOGLE_API_KEY"] = "Gemini key"
os.environ["GEMINI_MODEL"] = "gemini-3.1-flash-lite"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "Langsmith key"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-agents"

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from multi_agent import build_graph


def fresh_initial_state(topic: str) -> dict:
    return {
        "topic": topic,
        "messages": [],
        "notes": [],
        "subtopics": [],
        "draft": "",
        "critique": "",
        "approved": False,
        "revisions": 0,
        "max_revisions": 2,
    }

def human_review(graph, config, approve: bool, reason: str = "Rejected by human reviewer.") -> dict:
  
    if approve:
        return graph.invoke(None, config=config)

    current = graph.get_state(config)
    revisions = current.values.get("revisions", 0) + 1
    graph.update_state(
        config,
        {"approved": False, "critique": f"REVISE (human): {reason}", "revisions": revisions},
        as_node="critic",
    )
    return graph.invoke(None, config=config)


def save_graph_diagram(graph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(os.path.join(os.path.dirname(__file__), "graph.png"), "wb") as f:
            f.write(png_bytes)
        print("Saved graph diagram to graph.png")
    except Exception as exc:  # network-dependent (mermaid.ink) - non-fatal
        print(f"Could not render graph.png ({exc}); printing mermaid source instead:")
        print(graph.get_graph().draw_mermaid())


def main():
    store = InMemoryStore()

    # ---------------- RUN A: parallel branch ----------------
    print("\n########## RUN A: parallel fan-out branch ##########")
    graph_a = build_graph().compile(
        checkpointer=MemorySaver(), store=store, interrupt_before=["publish"]
    )
    save_graph_diagram(graph_a)

    config_a = {
        "configurable": {"thread_id": "session-A", "user_id": "alice"},
        "recursion_limit": 25,
    }
    graph_a.invoke(fresh_initial_state("Geothermal heat pumps, wave energy converters, and concentrated solar thermal plants"), config=config_a)
    state = graph_a.get_state(config_a)
    print("Subtopics:", state.values["subtopics"])
    print("Notes:", state.values["notes"])
    print("Revisions used:", state.values["revisions"])
    print("Paused before:", state.next)

    # ---------------- TIME TRAVEL ----------------
    print("\n---------- time travel over RUN A's history ----------")
    history_chronological = list(reversed(list(graph_a.get_state_history(config_a))))
    reject_checkpoint = None
    for snap in history_chronological:
        if snap.values.get("critique", "").upper().startswith("REVISE"):
            reject_checkpoint = snap
            break
    if reject_checkpoint:
        print("First rejection happened with critique:", reject_checkpoint.values["critique"])
        print("From that checkpoint, the graph was about to run:", reject_checkpoint.next)
        print(
            "To fork from here with a different outcome, you would call "
            "graph_a.update_state(reject_checkpoint.config, {...}) to inject a "
            "different value, then graph_a.invoke(None, config=reject_checkpoint.config)."
        )
    else:
        print("No rejection occurred in this run (the critic approved on the first pass).")

   # ---------------- human review of the breakpoint ----------------
    print("\n---------- human REJECTS the draft ----------")
    human_review(graph_a, config_a, approve=False, reason="Needs a stronger conclusion.")
    s = graph_a.get_state(config_a)
    print("Revisions after rejection:", s.values["revisions"])
    print("Critique on record:", s.values["critique"])
    print("Paused before (should be publish again):", s.next)

    print("\n---------- human APPROVES the revised draft ----------")
    human_review(graph_a, config_a, approve=True)
    print("RUN A finished. Final state.next:", graph_a.get_state(config_a).next)

    # ---------------- RUN B: memory-skip branch ----------------
    print("\n########## RUN B: same topic, new session -> memory-skip branch ##########")
    graph_b = build_graph().compile(
        checkpointer=MemorySaver(), store=store, interrupt_before=["publish"]
    )
    config_b = {
        "configurable": {"thread_id": "session-B", "user_id": "alice"},
        "recursion_limit": 25,
    }
    graph_b.invoke(fresh_initial_state("Solar power and Wind power"), config=config_b)
    print("Notes reused from long-term memory:", graph_b.get_state(config_b).values["notes"])
    graph_b.invoke(None, config=config_b)

    # ---------------- RUN C: single, non-parallel branch ----------------
    print("\n########## RUN C: single-subtopic topic -> non-parallel branch ##########")
    graph_c = build_graph().compile(
        checkpointer=MemorySaver(), store=store, interrupt_before=["publish"]
    )
    config_c = {
        "configurable": {"thread_id": "session-C", "user_id": "bob"},
        "recursion_limit": 25,
    }
    graph_c.invoke(fresh_initial_state("Nuclear power"), config=config_c)
    print("Subtopics:", graph_c.get_state(config_c).values["subtopics"])
    graph_c.invoke(None, config=config_c)


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Set GOOGLE_API_KEY before running (Gemini API key). Exiting.")
        raise SystemExit(1)
    main()
