from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from src.rag_pipeline import answer_question

class MaintenanceState(TypedDict):
    question: str
    answer: str
    sources: list[dict]


# Define a node
def answer_with_rag(state: MaintenanceState):
    """
    Call the existing RAG pipeline and store
    its answer and sources in the graph state.
    """

    result = answer_question(
        query=state["question"],
        top_k=4,
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }

builder = StateGraph(MaintenanceState)
builder.add_node("answer_with_rag", answer_with_rag)

# Connect the nodes
builder.add_edge(START, "answer_with_rag")
builder.add_edge("answer_with_rag", END)

maintenance_graph = builder.compile()