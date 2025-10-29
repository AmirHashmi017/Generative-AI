
# type: ignore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
text='''
Farmers are working hard in the fields, preparing the soil and planting seeds for 
the next season. The Sun was bright, and the air smeeled of earth and fresh grass.
The Pakistan Super League is one of the best Leagues inw world. Featuring multiple
international players doing top performances.

Terrorism in pakistan is reaching to a hazard level especially in KPK and Balochistan.
on recent bais ther are explosive attacks and things that are disturbing peace, economy.
'''

text_splitter= SemanticChunker(
    OpenAIEmbeddings(),breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

docs= text_splitter.split_text(text)
print(len(docs))
print(docs)