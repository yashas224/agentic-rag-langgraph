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

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

load_dotenv()

#  load documents
docs: list[list[Document]] = [WebBaseLoader(url).load() for url in urls]

# split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
doc_list = [item for doc in docs for item in doc]

docs_split = text_splitter.split_documents(documents=doc_list)
print(f"initial documents : {len(doc_list)} Final Chunks {len(docs_split)}")

#  Vectore Store Ingestion

vectorstore.from_documents(documents=docs_split)

retriver = vectorstore.as_retriever()
