from typing import Any, Dict

from langchain_core.documents import Document

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate_node(state: GraphState) -> Dict[str, Any]:
    docs = state.get("documents")
    question = state.get("question")
    generation = generation_chain.invoke(
        {"question": question, "context": "\n".join([doc.page_content for doc in docs])}
    )
    return {"generation": generation}
