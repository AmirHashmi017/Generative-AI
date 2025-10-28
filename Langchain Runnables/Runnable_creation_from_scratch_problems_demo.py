import random

# First Component
class MyLLM:
    def __init__(self):
        print("My LLM created")
    def predict(self,prompt):
        response=[
            'PSL is a cricket League',
            'Quaid-e-Azam born in Karachi',
            'Minar-e-Pakistan is in Lahore']
        return random.choice(response)

# Second Component
class MyPromptTemplate:
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables= input_variables
    
    def format(self,input_dict):
        return self.template.format(**input_dict)

# Third component
class MyLLMChain:
    def __init__(self,llm,prompt):
        self.llm=llm
        self.prompt=prompt
    
    def run(self,input_data):
        final_prompt= self.prompt.format(input_data)
        result= self.llm.predict(final_prompt)
        return result
        
        


template= MyPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['topic']
)

llm= MyLLM()

chain= MyLLMChain(llm,template)


print(chain.run({'length':'short','topic':'pakistan'}))