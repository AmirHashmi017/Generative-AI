from langgraph.store.memory import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

store= InMemoryStore()

# Creating Memory
namespace= ("user","u1")

store.put(namespace,"1",{"data":"User likes Pizza"})
store.put(namespace,"2",{"data":"User prefers Dark Mode"})

namespace2= ("user","u2")

store.put(namespace2,"1",{"data":"User likes Pasta"})
store.put(namespace2,"2",{"data":"User prefers Grid Style Navigation"})

# Retrieving all memories
items= store.search(namespace)
for item in items:
    print(item.value)

# Retreiving Particular Memory
item= store.get(namespace,"1")
print(item.value)

# Semantic Search
embedding_model= GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
sem_store= InMemoryStore(index={"embed":embedding_model,"dims":768})
sem_namespace= ("user","u1")

sem_store.put(sem_namespace,"1","User prefers concise answers over long explanations.")
sem_store.put(sem_namespace,"2","User likes examples in Python.")
sem_store.put(sem_namespace,"3","User usually works late at night.")
sem_store.put(sem_namespace,"4","User prefers dark mode in applications.")
sem_store.put(sem_namespace,"5","User is learning Machine Learning")
sem_store.put(sem_namespace,"6","User dislikes overly theoretical explanations")
sem_store.put(sem_namespace,"7","User prefers step-by-step reasoning.")
sem_store.put(sem_namespace,"8","User is based in Pakistan.")
sem_store.put(sem_namespace,"9","User likes real world analogies.")
sem_store.put(sem_namespace,"10","User prefers bullet points over paragraphs")

items= sem_store.search(sem_namespace,query="What is the user currently learning",limit=1)

print("User Learning: ")
for item in items:
    print(item.value)

items= sem_store.search(sem_namespace,query="What is the user's preferences",limit=3)

print("User Preferences: ")
for item in items:
    print(item.value)

