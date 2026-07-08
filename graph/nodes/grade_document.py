from typing import Any, Dict

from langchain_core.documents import Document

from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from graph.state import GraphState


def gradeDocNode(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    docs = state.get("documents")
    question = state.get("question")
    filtered_docs = []
    web_search: bool = False

    for d in docs:
        response: GradeDocument = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if response.isRelevant == 'yes':
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print(f"---GRADE: DOCUMENT NOT RELEVANT--- Reason \n{response.reason}")
            web_search = True
            continue

    return {"documents": filtered_docs, "web_search": web_search}
