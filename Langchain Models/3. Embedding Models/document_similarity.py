from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embedding= OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents=[
    'Babar Azam is a Pakistani cricketer known for his Classical Batting and cover drive',
    'M. Rizwan is a Pakistani cricketer known for his flick shot, Batting against spin and Wicket-Keeping',
    'Shaheen Afridi is a pakistani pacer known for his swing and wicket taking skills',
    'Haris Rauf is a pakistani speedstar known or his agressive and pacy bowling',
    'Fakhar Zaman is a pakistani batter known for hi agressive batting and power hitting'
]

query= 'Tell me about babar azam'

doc_embeddings= embedding.embed_documents(documents)
query_embedding= embedding.embed_query(query)

scores= cosine_similarity([query_embedding],doc_embeddings)[0]

index,query_score= sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print('Similarity Score is: ',query_score)
