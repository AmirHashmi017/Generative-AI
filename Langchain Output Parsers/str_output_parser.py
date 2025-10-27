from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion"
)

model= ChatHuggingFace(llm=llm)

prompt_template_1= PromptTemplate(
    template='''
Write a detailed report on topic {topic}''',
input_variables=['topic']
)

prompt_template_2= PromptTemplate(
    template='''
Write a 5 line summary on following text {text}''',
input_variables=['text']
)

# Without string output Parser

# prompt1= prompt_template_1.invoke({"topic":"Linear Regression"})

# result1= model.invoke(prompt1)

# prompt2= prompt_template_2.invoke({"text":result1.content})

# final_result= model.invoke(prompt2)

# print(final_result.content)

# With string output parser very compatible with chains

parser= StrOutputParser()

chain= prompt_template_1 | model | parser | prompt_template_2 | model | parser

result= chain.invoke("Linear Regression")

print(result)