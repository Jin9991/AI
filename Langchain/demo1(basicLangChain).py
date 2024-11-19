import os 
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith configuration (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

# Azure OpenAI Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_BASE")
deployment_name = "gpt-4"
api_version = "2024-10-01-preview"

# 1 step: make model
# Initialize the Azure OpenAI model
model = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
    deployment_name=deployment_name,
    api_version=api_version,
)

# 2 step: prepare prompt
# Create messages
messages = [
    SystemMessage(content='Please translate to Japanese'),
    HumanMessage(content='Hello, Where are you going?')
]

# Get the model response
#result = model.invoke(messages)
#print("Full result:")
#print(result)


# 3 step: make response analyzer
# Initialize the string parser
parser = StrOutputParser()

# Get just the content string
#return_str = parser.invoke(result.content)
#print("\nJust the translation:")
#print(return_str)

# 4 step: get chain
chain = model | parser

#5 use chain to do things
print(chain.invoke(messages))


# 5 step to use languagechain