from typing import Literal

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from openai import BaseModel
from pydantic import ConfigDict, Field

from ingestion import getVectoreStoreRetriever

load_dotenv()

model = ChatOpenAI(model="gpt-5.5", temperature=0)


class GradeDocument(BaseModel):
    """A Document Grader object with details."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="The user question")
    isRelevant: Literal["yes", "no"] = Field(
        description="Whether the document is relevant or not"
    )
    reason: str = Field(description="Why the document is relevant or not relevant")
    rating: int = Field(description="Relevancy rating out of 10")


model = model.with_structured_output(GradeDocument)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

humanMessage = "Retrieved document: \n\n {document} \n\n User question: {question}"
template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", humanMessage),
    ]
)

retrieval_grader = template | model

if __name__ == "__main__":
    # Testing python -m graph.chains.retrieval_grader
    # retriever = getVectoreStoreRetriever()
    # docs = retriever.invoke("Prompt engineering")
    # print(docs[0].page_content)

    # response: GradeDocument = retrieval_grader.invoke(
    #     {"question": "Agent Memory in LLM", "document": docs[0].page_content}
    # )
    # print(response)
    pass