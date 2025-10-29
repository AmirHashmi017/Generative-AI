# Can be used for splitting different documents like code, markdown etc. 
# Same as RecursiveCharacterTextSplotter but have different Separators
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter,Language

text='''
import random
from abc import ABC, abstractmethod
# We will make runnables by giving common structure to all classes because it is important that all classes have unified methods
# We will implement this by making an abstract class which enforce all classes t have that one method implemented

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass

# First Component
class MyLLM(Runnable):
    def __init__(self):
        print("My LLM created")
    def predict(self,prompt):
        response=[
            'PSL is a cricket League',
            'Quaid-e-Azam born in Karachi',
            'Minar-e-Pakistan is in Lahore']
        return random.choice(response)
    
    def invoke(self,prompt):
        response=[
            'PSL is a cricket League',
            'Quaid-e-Azam born in Karachi',
            'Minar-e-Pakistan is in Lahore']
        return random.choice(response)

# Second Component
class MyPromptTemplate(Runnable):
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables= input_variables
    
    def format(self,input_dict):
        return self.template.format(**input_dict)

    def invoke(self,input_dict):
        return self.template.format(**input_dict)

class RunnableConnector(Runnable):
    
    def __init__(self,runnable_list):
        self.runnable_list=runnable_list
    
    def invoke(self,input_data):
        for runnable in self.runnable_list:
            input_data= runnable.invoke(input_data)
        return input_data

prompt1= MyPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2= MyPromptTemplate(
    template='Explain the following joke {joke}',
    input_variables=['joke']
)

llm= MyLLM()

chain1 = RunnableConnector([prompt1,llm])

chain2= RunnableConnector([prompt2,llm])

final_chain= RunnableConnector([chain1,chain2])

result= final_chain.invoke({'topic':'cricket'})
print(result)

'''

splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks= splitter.split_text(text)

print(len(chunks))
print(chunks[1])