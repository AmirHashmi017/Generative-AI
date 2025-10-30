from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.schema import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="Eating fruits daily improves immunity."),
    Document(page_content="Regular exercise keeps the heart healthy."),
    Document(page_content="Drinking water is essential for hydration."),
    Document(page_content="Sleeping well boosts mental health."),
    Document(page_content="Meditation reduces stress and anxiety."),
    Document(page_content="Vaccines protect against serious diseases."),
    Document(page_content="Solar systems helps balancing electricity load."),
    Document(page_content="Washing hands prevents infections."),
    Document(page_content="LangChain helps build AI applications."),
    Document(page_content="LangChain works with LLMs easily."),
    Document(page_content="LangChain supports vector databases."),
]


embedding_model= OpenAIEmbeddings()

# It is both creating vector store and adding documents
vector_store= FAISS.from_documents(
    documents=documents,
    embedding= embedding_model
)

multiquery_retriever= MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k":5}),
    llm= ChatOpenAI(model='gpt-5')
 )

query= "How to improve energy levels and maintain balance?"
results= multiquery_retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n -- Result {i+1} --")
    print(f"Content \n {doc.page_content}")