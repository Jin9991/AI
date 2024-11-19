import os
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import chat_agent_executor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith configuration (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Demo5"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["TAVILY_API_KEY"] = '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'

# Azure OpenAI Configuration
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize the Azure OpenAI model with retry configuration
model = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
    deployment_name="gpt-4",  # Example deployment, replace with actual deployment name
    api_version=api_version,
    max_retries=5,
    request_timeout=30
)


# Withou agetn AI will tell you search by yourself cause she don't have the information
# result = model.invoke([HumanMessage(content='How about Tokyo recent weather?')])
# print(result)

# Langachain has tooll build inside, "Tavily" can used to be the search engigne tool
search = TavilySearchResults(max_results=2) #max_results like the name it only return 2 search results
#print(search.invoke('How about Tokyo recent weather?'))

#Let model bind tools to use search engine
tools = [search]
# model_with_tool = model.bind_tools([tools])

# First resp shows LLM will answer the question base on their knowledge, resp2 shows that LLM is smart enogugh they will see if they need to use search or not
# resp = model_with_tool.invoke([HumanMessage(content="Where are you from?")])

# print(f'Model_Result_Content: {resp.content}')
# print(f'Tools_Result_Content: {resp.tool_calls}')


# resp2 = model_with_tool.invoke([HumanMessage(content='How about Tokyo recent weather?')])

# print(f'Model_Result_Content: {resp2.content}')
# print(f'Tools_Result_Content: {resp2.tool_calls}')

# make an agent

agent_excutor = chat_agent_executor.create_tool_calling_executor(model, tools)

# resp = agent_excutor.invoke({'messages': [HumanMessage(content='Where are you from?')]})
# print(resp['messages'])

resp2 = agent_excutor.invoke({'messages': [HumanMessage(content='How about Tokyo recent weather?')]})
# print(resp2['messages'])
print(resp2['messages'][2].content)