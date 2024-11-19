import os 
import time
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'

# Azure OpenAI Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_BASE")
deployment_name = "gpt-4"
api_version = "2024-10-01-preview"

# 1. Initialize the Azure OpenAI model with retry mechanism
model = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
    deployment_name=deployment_name,
    api_version=api_version,
    temperature=0.7,
    request_timeout=30,
    max_retries=3,
    # Add rate limiting
    frequency_penalty=0,
    presence_penalty=0
)

# 2. Prepare prompt
messages = [
    SystemMessage(content='Please translate to Japanese'),
    HumanMessage(content='Hello, Where are you going?')
]

# 3. Initialize the string parser
parser = StrOutputParser()

# 4. Create the chain
chain = model | parser

# 5. Function to execute chain with retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda _: "Error: Rate limit exceeded. Please try again later."
)
def execute_chain(input_messages):
    try:
        result = chain.invoke(input_messages)
        return result
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

# 6. Execute the chain with error handling
try:
    print("Sending request...")
    result = execute_chain(messages)
    print("\nTranslation result:")
    print(result)
    
except Exception as e:
    print(f"Final error: {str(e)}")