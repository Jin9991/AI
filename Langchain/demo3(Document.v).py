import os
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith configuration (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Demo4"
os.environ["LANGCHAIN_API_KEY"] = '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'

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

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",  # Replace with your actual deployment name
    openai_api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=openai_api_key,
)

# Prepare test data
documents = [
    Document(
        page_content="Oda is known by Onepiece",
        metadata={"source": "Jump", "type": "manga"},
    ),
    Document(
        page_content="Tatsuki is known by ChainsawMan",
        metadata={"source": "Jump", "type": "manga"},
    ),
    Document(
        page_content="Miura is known by Berserk",
        metadata={"source": "YoungAnimal", "type": "manga"},
    ),
    Document(
        page_content="Aoyama is known by DetectiveConan",
        metadata={"source": "Sunday", "type": "manga"},
    ),
    Document(
        page_content="Takahashi is known by Ranma1/2",
        metadata={"source": "Sunday", "type": "manga"},
    ),
]

# Instantiate a vector database with embeddings
vector_store = Chroma.from_documents(documents, embedding=embeddings)

# Search engine (Runnable): Retrieve the most similar document
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

# Test retriever
# print(retriever.batch(['Onepiece', 'Ranma']))

# Prompt template
message = """
Answer the question using the provided data:
{question}
Provided data:
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

# RunnablePassthrough() allows passing user's question to prompt and model
chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model

resp = chain.invoke('Please introduce about Takahashi')

print(resp.content)