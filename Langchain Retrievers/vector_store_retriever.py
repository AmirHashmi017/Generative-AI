from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.schema import Document
from dotenv import load_dotenv

load_dotenv()

documents=[
    Document(page_content="Langchain helps developer build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM-based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="OpenAI provides powerful Embedding Models")
]

embedding_model= OpenAIEmbeddings()

# It is both creating vector store and adding documents
vector_store= Chroma.from_documents(
    documents=documents,
    embedding= embedding_model,
    collection_name='my_collection'
)

# Convert vector store into retriever
retriever= vector_store.as_retriever(search_kwargs={"k":2})

query= "What is chroma used for?"
results= retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n -- Result {i+1} --")
    print(f"Content \n {doc.page_content}")