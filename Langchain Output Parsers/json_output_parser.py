from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion"
)

model= ChatHuggingFace(llm=llm)
parser= JsonOutputParser()

prompt_template= PromptTemplate(
    template='''
Git me the name, age, city of a fictional person \n {format_instruction}''',
input_variables=[],
partial_variables= {'format_instruction':parser.get_format_instructions}
)

prompt =prompt_template.format()

result= model.invoke(prompt)
print(result.content)