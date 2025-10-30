from langchain_classic.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.schema import Document
from dotenv import load_dotenv

load_dotenv()

doc1= Document(
    page_content="Babar Azam is a prolific right-handed batsman, he is widely regarded as one of the world's premier stroke-makers and a consistent top-ranked player.",
    metadata={"team":"Peshawar Zalmi"}
)

doc2= Document(
    page_content="Shaheen Afridi is a towering left-arm fast bowler who generates significant pace and bounce, making him a major wicket-taking threat with the new ball.",
    metadata={"team":"Lahore Qalanadars"}
)

doc3= Document(
    page_content="M.Rizwan is an energetic right-handed wicket-keeper-batsman, recognized for his explosive hitting in T20Is and his valuable contributions in the middle order in all formats.",
    metadata={"team":"Multan Sultans"}
)

doc4= Document(
    page_content="Shadab Khan is a dynamic leg-spinning all-rounder, he is crucial to the team for his ability to break partnerships with his wrist spin and provide aggressive lower-order batting.",
    metadata={"team":"Islamabad United"}
)

doc5= Document(
    page_content="Sarfaraz Ahmed is a spirited wicket-keeper-batsman and a former captain who led Pakistan to a historic victory in the 2017 ICC Champions Trophy.",
    metadata={"team":"Quetta Gladiators"}
)

docs= [doc1,doc2,doc3,doc4,doc5]

# Create Vector Store
vector_store= Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="my_chroma_db",
    collection_name="cricketers"
)

# Add Document
vector_store.add_documents(docs)

# View Documents
documents= vector_store.get(include=['embeddings','documents','metadatas']) # Telling in include what what we want to see
print(documents)

# Search Document
result= vector_store.similarity_search(
    query="Who among these are bowlers?",
    k=2 # Specify hw many similar you want to get
)

print(f"The bowler are: ${result}")

# Search Documents and also get similarity scores
result= vector_store.similarity_search_with_score(
    query="Who among these are bowlers?",
    k=2 # Specify hw many similar you want to get
)

print(f"The bowlers with Score are: ${result}")

# Meta-data based filtering
result= vector_store.similarity_search_with_score(
    query="",
    filter={'team':'Peshawar Zalmi'}
)

print(result)

# Update Document

update_doc1= Document(
    page_content="Babar Azam is known for his elegant technique and holds numerous records, including becoming one of Pakistan's most successful captains across all formats.",
    metadata={"team":"Peshawar Zalmi"}
)

vector_store.update_document(document_id="3c981ebb-4364-46c5-b51f-bfd7c20c0275",document=update_doc1)

# Delete Document
vector_store.delete(ids=['f47e0b4b-a3b9-4bd6-a39a-c16986e7ed5a'])

documents= vector_store.get(include=['embeddings','documents','metadatas']) # Telling in include what what we want to see
print(documents)