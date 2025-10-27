from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it",
                         task= "text-generation")

model= ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str= Field(description="Name of the Person")
    age: int= Field(gt=18,description="Age of the person")
    city: str = Field(description="Name of city the person belongs to")

parser= PydanticOutputParser(pydantic_object=Person)

prompt_template= PromptTemplate(
    template='''
Give me the name, age, city of a fictional {place} person \n {format_instruction}''',
input_variables=["place"],
partial_variables= {'format_instruction':parser.get_format_instructions()}
)

prompt= prompt_template.invoke({"place":"Arab"})

result= model.invoke(prompt)
final_result= parser.parse(result.content)
print(final_result)