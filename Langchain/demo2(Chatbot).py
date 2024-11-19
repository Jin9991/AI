import os 
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith configuration (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Demo3Chatbot"
os.environ["LANGCHAIN_API_KEY"] = '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'

# Azure OpenAI Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_BASE")
deployment_name = "gpt-4"
api_version = "2024-10-01-preview"

# Initialize the Azure OpenAI model with retry configuration
model = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
    deployment_name=deployment_name,
    api_version=api_version,
    # Add retry configuration
    max_retries=5,
    request_timeout=30
)

# Create prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", 'You are very gentleman helper. Use {language} to answer the question you can answer'),
    MessagesPlaceholder(variable_name='my_msg')
])

# Create the chain
chain = prompt_template | model 

# Save Conversation History
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'
)

config = {'configurable': {'session_id': 'test123'}}

# Helper function for making requests with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
def make_request(message_func, *args, **kwargs):
    return message_func(*args, **kwargs)

try:
    # First conversation
    print("Starting first conversation...")
    resp1 = make_request(do_message.invoke,
        {
            'my_msg': [HumanMessage(content="Hello, I'm Jin")],
            'language': 'English'
        },
        config=config
    )
    print("Response 1:", resp1.content)
    
    # Add delay between requests
    time.sleep(10)  # Wait 10 seconds between requests
    
    # Second conversation
    print("\nStarting second conversation...")
    resp2 = make_request(do_message.invoke,
        {
            'my_msg': [HumanMessage(content="Question: What is my name?")],
            'language': 'English'
        },
        config=config
    )
    print("Response 2:", resp2.content)
    
    # Add delay before streaming
    time.sleep(10)  # Wait 10 seconds before starting stream
    
    # Third conversation (streaming)
    print("\nStarting streaming conversation...")
    for chunk in do_message.stream(
        {
            'my_msg': [HumanMessage(content='Please tell me a joke with my name')],
            'language': 'English'
        },
        config=config
    ):
        print(chunk.content, end='-')
        
except Exception as e:
    print(f"\nAn error occurred: {str(e)}")