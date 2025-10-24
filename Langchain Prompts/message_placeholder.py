from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
import re

# Chat Template
chat_template= ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

# Load Chat History
chat_history=[]
with open('chat_history.txt') as file:
    for line in file.readlines():
        if line.startswith('HumanMessage'):
            match = re.search(r'content="(.+?)"', line)
            if match:
                chat_history.append(HumanMessage(content=match.group(1))) 
                
        elif line.startswith('AIMessage'):
            match = re.search(r'content="(.+?)"', line)
            if match:
                chat_history.append(AIMessage(content=match.group(1))) 


# Create prompt
prompt= chat_template.invoke({'chat_history':chat_history,'query':'Where is my refund'})

print(prompt)