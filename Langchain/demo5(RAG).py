import os
import bs4
import diskcache as dc
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define a retry decorator for handling rate limits
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=3, min=20, max=120),  # Increased backoff significantly
    stop=stop_after_attempt(5)
)
def rate_limited_completion(chain, input_text):
    return chain.invoke({'input': input_text})


# LangSmith configuration (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Demo5_RAG"
os.environ["LANGCHAIN_API_KEY"] = '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'

# Azure OpenAI Configuration
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize the Azure OpenAI model with modified settings
model = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
    deployment_name="gpt-4",  # Ensure correct model and deployment
    api_version=api_version,
    max_retries=3,
    request_timeout=30,
    temperature=0.7,
    max_tokens=200,  # Limit response tokens to reduce load
    frequency_penalty=0,
    presence_penalty=0
)


# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",  # Replace with your actual deployment name
    openai_api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=openai_api_key,
)

# Set up caching to store responses locally
cache = dc.Cache("/tmp/langchain_cache")

def rate_limited_completion_with_cache(chain, input_text):
    # Check cache first
    if input_text in cache:
        return cache[input_text]
    
    # Fetch response and store in cache
    response = rate_limited_completion(chain, input_text)
    cache[input_text] = response
    return response


# Load data from online blog
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )    
)

docs = loader.load()

# print(len(docs))
# print(docs)

#2 Big data split data

# test text: text = "Hello, Who are you? thank, I'm fine. There was 3 apple trees, 2 banana trees, and 1 Watermelon tree there. Hopefully, you will like those fruits. "

spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)

splits = spliter.split_documents(docs)

#3 Storage
vectoreStore = Chroma.from_documents(documents = splits, embedding = AzureOpenAIEmbeddings())

# 4 Create Retriver
retriver = vectoreStore.as_retriever()

# 5 collabolate

# make an question prompt
system_prompt = """ You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Use three sentences maximum and keep the answer concise. \n

{context}
"""

prompt = ChatPromptTemplate.from_messages( # question & answer Record
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# get chain
chain1 = create_stuff_documents_chain(model, prompt) # this is question chain

#chain2 = create_retrieval_chain(retriver, chain1) # include retriver, first do search and then do answer

# try:
#     resp = rate_limited_completion(chain2, "What is Task Decomposition?")
#     print(resp['answer'])
# except RateLimitError:
#     print("The rate limit was reached. Please try again later.")
# except Exception as e:
#     print(f"Failed after multiple retries: {str(e)}")
    

# child chain prompt temple
contextualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is ."""

retriver_history_template = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt ),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}"),
    ]
)

#make an child chain 
history_chain = create_history_aware_retriever(model, retriver, retriver_history_template)

# save Q&A history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# make an parent chain 
chain = create_retrieval_chain(history_chain, chain1)

result_chain = RunnableWithMessageHistory(
    chain, 
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)


# first conversation
resp1 = result_chain.invoke(
    {'input': 'What is Task Decomposition?")'},
    config={'configurable':{'session_id': 'hi123'}}
)

print(resp1['answer'])

# second conversation
resp2 = result_chain.invoke(
    {'input': 'What are common ways of doing it?")'},
    config={'configurable':{'session_id': 'hi123'}}
)

print(resp2['answer'])