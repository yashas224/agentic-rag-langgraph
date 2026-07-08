import langchain
import langsmith
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import Client

from ingestion import getVectoreStoreRetriever

print(langchain.__version__)
print(langsmith.__version__)

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
load_dotenv()

model = ChatOpenAI(model="gpt-5.5", temperature=0)

generation_chain = prompt | model | StrOutputParser()


if __name__ == "__main__":
    # Testing
    # retriever = getVectoreStoreRetriever()
    # docs = retriever.invoke("What is Prompt engineering")
    # print(docs[0].page_content)
    # response = generation_chain.invoke({
    #     "question":"What is Prompt engineering",
    #     "context" : "\n".join([doc.page_content for doc in docs])
    # })

    # print(f"Response: {response}")
    pass
