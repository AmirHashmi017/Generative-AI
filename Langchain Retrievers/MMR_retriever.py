from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.schema import Document
from dotenv import load_dotenv

load_dotenv()

documents=[
    Document(page_content="Langchain makes it easy to work with LLMs."),
    Document(page_content="Langchain helps developer build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM-based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="OpenAI provides powerful Embedding Models"),
    Document(page_content="MMR is an information retrieval algorithm designed to reduce redundancy in the retrieved results while maintaining high relevance to the query."),
    Document(page_content="Langchain support schroma and FAISS."),
]

embedding_model= OpenAIEmbeddings()

# It is both creating vector store and adding documents
vector_store= FAISS.from_documents(
    documents=documents,
    embedding= embedding_model
)

# Convert vector store into retriever
retriever= vector_store.as_retriever(search_type="mmr", search_kwargs={"k":3,"lambda_mult":0.5})
# Lambda Mult it ranges from 0 to 1 more it is close to 0 more diverse results we will get

query= "What is Langchain?"
results= retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n -- Result {i+1} --")
    print(f"Content \n {doc.page_content}")