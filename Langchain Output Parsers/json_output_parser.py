from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


model= ChatOpenAI(model='gpt-5')
parser= JsonOutputParser()

prompt_template= PromptTemplate(
    template='''
Git me the name, age, city of a fictional person \n {format_instruction}''',
input_variables=[],
partial_variables= {'format_instruction':parser.get_format_instructions()}
)

chain= prompt_template | model | parser

result= chain.invoke({})
print(result)