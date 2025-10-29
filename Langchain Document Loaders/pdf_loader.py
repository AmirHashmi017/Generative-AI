from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader('Amir Resume.pdf')

docs= loader.load()

print(docs)
print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)

# PyPDF Loader is not very good in extracting Scanned Images so for that 
# we use different PDF Loaders