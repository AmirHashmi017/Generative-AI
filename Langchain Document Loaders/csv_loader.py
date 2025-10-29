from langchain_community.document_loaders import CSVLoader

loader= CSVLoader('Churn_Modelling.csv')

docs= loader.load()

print(len(docs)) # there will be a docs for every row

print(docs[0])