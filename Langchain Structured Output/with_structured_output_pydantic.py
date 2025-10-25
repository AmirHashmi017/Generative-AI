from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional,Literal

load_dotenv()

model=ChatOpenAI(model='gpt-5')

review= """I've been using the XPhone Pro for three weeks now, and I have mixed feelings about it. 

The display quality is absolutely stunning - the 6.7-inch AMOLED screen with 120Hz refresh rate 
makes scrolling incredibly smooth and watching videos is a pleasure. The colors are vibrant and 
the brightness is excellent even in direct sunlight.

Performance-wise, the processor handles everything I throw at it. Gaming runs smoothly, 
multitasking is seamless, and apps open instantly. The 12GB RAM ensures nothing slows down.

The camera system is impressive. The 108MP main sensor captures incredible detail in daylight, 
and the night mode produces clean shots even in low light. The ultra-wide lens is great for 
landscapes, though the zoom quality could be better.

However, the battery life is disappointing. With moderate use, I barely make it through a full 
day. I find myself charging twice daily, which is frustrating for a flagship device. The 
fast charging helps, but it shouldn't be necessary this often.

The software experience is where things get problematic. The UI is cluttered with bloatware - 
pre-installed apps I'll never use that can't be uninstalled. The interface feels outdated 
compared to competitors, and there are too many duplicate apps for basic functions.

The build quality is premium with a glass back and metal frame, but it's a fingerprint magnet. 
The phone is also quite heavy and slippery, making it uncomfortable for extended use.

Price-wise, at $1200, it feels overpriced given the battery and software issues. Competitors 
offer similar specs for $200-300 less with better optimization.

In summary: great hardware held back by poor software choices and battery problems."""

# Schema of Pydantic

class Review(BaseModel):
    key_themes: list[str]= Field(description="Write down all the key themes discussed in review")
    summary: str= Field(description="A brief summary of review")
    sentiment: Literal['pos','neg','nor']= Field(description="Return sentiment of review either positive, negative or normal")
    pros: Optional[list[str]]= Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]]= Field(default=None, description="Write down all the cons inside a list")

structured_model= model.with_structured_output(Review)

result= structured_model.invoke(review)

print(result)
print(result.summary)