"""
LangGraph workflow definition for InkGraph.

Pipeline:
    planner → search → writer → fact_checker → reviewer
                        ▲             │           │
                        ├─────────────┴───────────┤ (needs revision?)
                        │                         ▼
                        │                   tone_optimizer → [INTERRUPT: human]
                        │                                          │
                        └───────────── (changes?) ─────────────────┘
                                                                   ▼
                                                                  END
"""

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.nodes import (
    human_gate_node,
    planner_node,
    search_node,
    writer_node,
    fact_checker_node,
    reviewer_node,
    tone_optimizer_node,
)


class DocumentState(TypedDict):
    document_id: str
    prompt: str
    outline: dict | None
    draft: str | None
    review_notes: list[str]
    needs_revision: bool
    review_cycle: int
    human_decision: str | None  # "approved" | "changes" | None (pending)
    word_limit: int | None
    writing_style: str
    search_results: str | None



# Routing functions


def route_after_fact_check(state: DocumentState) -> Literal["writer", "reviewer"]:
    """Send back to writer if fact-checker detected errors, else pass to reviewer."""
    return "writer" if state.get("needs_revision", False) else "reviewer"


def route_after_review(state: DocumentState) -> Literal["writer", "tone_optimizer"]:
    """Send back to writer if reviewer detected errors, else pass to tone optimizer."""
    return "writer" if state.get("needs_revision", False) else "tone_optimizer"


def route_after_human(state: DocumentState) -> Literal["writer", "__end__"]:
    """After human decision: restart writer loop or finish."""
    return "writer" if state.get("human_decision") == "changes" else END


# Graph builder


def build_workflow():
    """
    Compile the InkGraph LangGraph workflow with a MemorySaver checkpointer.
    """
    graph = StateGraph(DocumentState)

    # Register all nodes
    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("writer", writer_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("tone_optimizer", tone_optimizer_node)
    graph.add_node("human", human_gate_node)

    # Edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "writer")
    graph.add_edge("writer", "fact_checker")
    
    graph.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        {"writer": "writer", "reviewer": "reviewer"},
    )
    
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", "tone_optimizer": "tone_optimizer"},
    )
    
    graph.add_edge("tone_optimizer", "human")
    
    graph.add_conditional_edges(
        "human",
        route_after_human,
        {"writer": "writer", END: END},
    )

    checkpointer = MemorySaver()

    return graph.compile(interrupt_before=["human"], checkpointer=checkpointer)
