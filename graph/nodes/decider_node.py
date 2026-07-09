from langchain_classic.schema import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from graph.state import GraphState
from ingestion import getVectoreStore


def decider_node(state: GraphState):
    """Node that decides the next state of  after receiving user question based on best similarity score
    low score is better match
    """
    print("Deciding the route")
    decission=""
    thresholdScore = 0.35
    question = state["question"]
    print(f"User Question {question}")

    vectoreStore: VectorStore = getVectoreStore()
    response: list[tuple[Document, float]] = vectoreStore.similarity_search_with_score(
        k=1, query=question
    )
    if response:
        print(f"Best matching vectore document score : {response[0][1]}")

    if response and response[0][1] < thresholdScore:
        
        decission = "vectorstore-search"
    else:
        decission = "web-search"
    
    print(f"Decision :{decission}")

    return {"initialDecision": decission}


if __name__ == "__main__":
    # Testing python -m graph.nodes.decider_node
    vectoreStore: VectorStore = getVectoreStore()
    response: list[tuple[Document, float]] = vectoreStore.similarity_search_with_score(
        k=1, query="What is Machine Learning way of training  LLM ?"
    )
    print(response)
