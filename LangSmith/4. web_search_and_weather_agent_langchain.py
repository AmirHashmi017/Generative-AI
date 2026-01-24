from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from dotenv import load_dotenv
import os

os.environ['LANGCHAIN_PROJECT']="Web Search and Weather Agent"
load_dotenv()
WEATHERSTACK_API_KEY= os.getenv("WEATHERSTACK_API_KEY")

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """"This tool takes location and gives it's temprature in output"""
  url= f"https://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"

  response = requests.get(url)

  return response.json()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)



response = agent_executor.invoke({"input": "Identify the birthplace city of Quaid-e-Azam and give its current temperature."})
print(response)

print(response['output'])