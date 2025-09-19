# from __future__ import annotations

# import uuid
# from functools import lru_cache
# from typing import Any, Dict, Iterable, List, Optional, TypedDict

# from langgraph.graph import END, START, StateGraph

# from agents import analyst, copywriter, ideator, manager
# from agents._llm import chat
# from memory.rag import init_project


# class CampaignState(TypedDict, total=False):
#     run_id: str
#     brief: str
#     model: str
#     project_title: str
#     plan: str
#     icp: str
#     concepts: str
#     copy: str
#     summary: str
#     artifacts: List[str]
#     board: Dict[str, Dict[str, Any]]
#     todo: List[Dict[str, Any]]


# @lru_cache(maxsize=1)
# def _compiled_graph():
#     graph = StateGraph(CampaignState)
#     graph.add_node("manager_plan", manager.plan)
#     graph.add_node("analyst_icp", analyst.icp)
#     graph.add_node("ideator_concepts", ideator.concepts)
#     graph.add_node("copywriter_texts", copywriter.texts)
#     graph.add_node("manager_summary", manager.assemble)

#     graph.add_edge(START, "manager_plan")
#     graph.add_edge("manager_plan", "analyst_icp")
#     graph.add_edge("analyst_icp", "ideator_concepts")
#     graph.add_edge("ideator_concepts", "copywriter_texts")
#     graph.add_edge("copywriter_texts", "manager_summary")
#     graph.add_edge("manager_summary", END)

#     return graph.compile()


# def build_graph():
#     """Return a compiled LangGraph workflow for the campaign."""
#     return _compiled_graph()


# def prepare_initial_state(
#     brief: str,
#     *,
#     model: Optional[str] = None,
#     project_title: Optional[str] = None,
#     extra_documents: Optional[Dict[str, str]] = None,
# ) -> CampaignState:
#     run_id = f"run-{uuid.uuid4().hex[:8]}"
#     init_project(run_id, brief, extra_documents)

#     return CampaignState(
#         run_id=run_id,
#         brief=brief,
#         model=model or DEFAULT_MODEL,
#         project_title=project_title or "Маркетинговая кампания",
#         artifacts=[],
#         board={},
#     )


# def run_campaign(state: CampaignState) -> CampaignState:
#     graph = build_graph()
#     return graph.invoke(state)


# def stream_campaign(state: CampaignState) -> Iterable[Dict[str, Any]]:
#     graph = build_graph()
#     yield from graph.stream(state)


# __all__ = [
#     "CampaignState",
#     "build_graph",
#     "prepare_initial_state",
#     "run_campaign",
#     "stream_campaign",
# ]
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import analyst, copywriter, ideator, manager
from agents._llm import chat
from memory.rag import init_project


class CampaignState(TypedDict, total=False):
    run_id: str
    brief: str
    model: Optional[str]
    project_title: str
    plan: str
    icp: str
    concepts: str
    copy: str
    summary: str
    artifacts: List[str]
    board: Dict[str, Dict[str, Any]]
    todo: List[Dict[str, Any]]


@lru_cache(maxsize=1)
def _compiled_graph():
    graph = StateGraph(CampaignState)
    graph.add_node("manager_plan", manager.plan)
    graph.add_node("analyst_icp", analyst.icp)
    graph.add_node("ideator_concepts", ideator.concepts)
    graph.add_node("copywriter_texts", copywriter.texts)
    graph.add_node("manager_summary", manager.assemble)

    graph.add_edge(START, "manager_plan")
    graph.add_edge("manager_plan", "analyst_icp")
    graph.add_edge("analyst_icp", "ideator_concepts")
    graph.add_edge("ideator_concepts", "copywriter_texts")
    graph.add_edge("copywriter_texts", "manager_summary")
    graph.add_edge("manager_summary", END)

    return graph.compile()


def build_graph():
    """Return a compiled LangGraph workflow for the campaign."""
    return _compiled_graph()


def prepare_initial_state(
    brief: str,
    *,
    model: Optional[str] = None,
    project_title: Optional[str] = None,
    extra_documents: Optional[Dict[str, str]] = None,
) -> CampaignState:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    init_project(run_id, brief, extra_documents)

    return CampaignState(
        run_id=run_id,
        brief=brief,
        model=model,  # ⚡️ убрали DEFAULT_MODEL
        project_title=project_title or "Маркетинговая кампания",
        artifacts=[],
        board={},
    )


def run_campaign(state: CampaignState) -> CampaignState:
    graph = build_graph()
    return graph.invoke(state)


def stream_campaign(state: CampaignState) -> Iterable[Dict[str, Any]]:
    graph = build_graph()
    yield from graph.stream(state)


__all__ = [
    "CampaignState",
    "build_graph",
    "prepare_initial_state",
    "run_campaign",
    "stream_campaign",
]
