from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader('Amir Resume.pdf')

docs= loader.load()

splitter= CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result= splitter.split_documents(docs)

print(result)