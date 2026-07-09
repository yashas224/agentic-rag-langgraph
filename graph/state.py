import operator

from langchain.messages import AnyMessage
from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph, add_messages
from typing_extensions import Annotated, TypedDict


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: list[Document]
    initialDecision: str
