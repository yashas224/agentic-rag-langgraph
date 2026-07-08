from typing import Literal
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ingestion import getVectoreStoreRetriever
from graph.chains.generation import generation_chain

load_dotenv()

model = ChatOpenAI(model="gpt-5.5", temperature=0)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: Literal["yes", "no"] = Field(
        description="Whether the answer is grounded in the provided documents"
    )

model = model.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


template = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader : RunnableSequence = template | model

if __name__ == "__main__":
    # Testing python -m graph.chains.hallucination_grader
    retriever = getVectoreStoreRetriever()
    question = "What is Prompt engineering"
    docs = retriever.invoke(input=question)
    response = generation_chain.invoke(
        {"question": question, "context":docs}
    )
    print(f"Answer: {response}")


    grader : GradeHallucinations =hallucination_grader.invoke({
        "documents": docs,
        "generation":response
    })

    print(grader)
