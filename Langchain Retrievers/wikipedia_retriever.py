from langchain_community.retrievers import WikipediaRetriever

retriever= WikipediaRetriever(top_k_results=2,lang="en")

query= "The geopolitical history of Pakistan and India from the perspective of a Chinese"

docs= retriever.invoke(query)

for i,doc in enumerate(docs):
    print(f"\n -- Result {i+1} --")
    print(f"Content \n {doc.page_content}")
    