from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory="./.chroma_db",
    embedding_function=embeddings,
    collection_name="index-rag-chroma",
)


def getVectoreStore():
    return vectorstore


load_dotenv()
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def loadDocuments():
    #  load documents
    docs: list[list[Document]] = [WebBaseLoader(url).load() for url in urls]
    return docs


def chunkAndSplitDocuments(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    doc_list = [item for doc in docs for item in doc]

    docs_split = text_splitter.split_documents(documents=doc_list)
    print(f"initial documents : {len(doc_list)} Final Chunks {len(docs_split)}")
    return docs_split

    #  Vectore Store Ingestion


def ingestIntoVectoreStore(docs_split):
    vectorstore = Chroma.from_documents(
        documents=docs_split,
        collection_name="index-rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory="./.chroma_db",
    )


def getVectoreStoreRetriever():
    return Chroma(
        collection_name="index-rag-chroma",
        persist_directory="./.chroma_db",
        embedding_function=OpenAIEmbeddings(),
    ).as_retriever()


if __name__ == "__main__":
    docs = loadDocuments()
    docs_split = chunkAndSplitDocuments(docs)
    ingestIntoVectoreStore(docs_split)
