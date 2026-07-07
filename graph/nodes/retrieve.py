from typing import Any, Dict

from langchain_core.documents import Document
from state import GraphState

from ingestion import getVectoreStoreRetriever


def retrieveNode(state: GraphState) -> Dict[str, Any]:
    print("retrieveNode Execution")
    question = state.get("question")
    retriever = getVectoreStoreRetriever()
    relevant_doc_list: list[Document] = retriever.invoke(question)
    return {"documents": relevant_doc_list}
