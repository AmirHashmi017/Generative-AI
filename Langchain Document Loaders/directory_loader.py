from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader= DirectoryLoader(
    path='books',
    glob='*.pdf', # All PDFs in books folder
    loader_cls=PyPDFLoader
)

# Code with Load (Load all at once then operation)
# docs= loader.load()

# for doc in docs:
#     print(doc.page_content)

# Code with Lazy Load (Loads one at time when required) is available with all document loader.
docs= loader.lazy_load()

for doc in docs:
    print(doc.page_content)