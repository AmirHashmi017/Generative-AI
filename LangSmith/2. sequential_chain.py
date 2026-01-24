from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config= {
    'run_name':'sequential_chain',
    'tags': ['llm_app','report generation','summarization'],
    'metadata': {'model1':'gemini-2.5-flash','parser':'stroutputparser'}
}

result = chain.invoke({'topic': 'Unemployment in Pakistan'},config=config)

print(result)