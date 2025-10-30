from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.schema import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="Photosynthesis covert light into food like use solar energy, and LangChain helps build AI applications that can track health data."),
    Document(page_content="Regular exercise keeps the heart healthy, while LangChain works with LLMs to provide smart fitness suggestions."),
    Document(page_content="Meditation reduces stress and anxiety, and LangChain supports vector databases for storing health-related information."),
]



embedding_model= OpenAIEmbeddings()

# It is both creating vector store and adding documents
vector_store= FAISS.from_documents(
    documents=documents,
    embedding= embedding_model
)
base_retriever= vector_store.as_retriever(search_kwargs={"k":1})
llm= ChatOpenAI(model='gpt-5')
compressor= LLMChainExtractor.from_llm(llm)

contextual_compression_retriever= ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor= compressor
 )

query= "Tell me about Photosynthesis?"
results= contextual_compression_retriever.invoke(query)
print(results)
for i,doc in enumerate(results):
    print(f"\n -- Result {i+1} --")
    print(f"Content \n {doc.page_content}")