from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from graph.state import GraphState


def webSearchNode(state: GraphState) -> Dict[str, Any]:
    tool = TavilySearch(max_results=3, topic="general")
    question = state.get("question")
    filtered_docs = state.get("documents")

    response = tool.invoke({"query": question}).get("results")

    final_search_result = "\n\n".join([part["content"] for part in response])
    web_result_doc = Document(page_content=final_search_result)
    if filtered_docs:
        filtered_docs.append(web_result_doc)
    else:
        filtered_docs = [web_result_doc]

    return {"documents": filtered_docs}


if __name__ == "__main__":
    webSearchNode(state={"question": "agent memory", "documents": None})
