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
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Demo4"
os.environ["LANGCHAIN_API_KEY"] = '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'

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5)
)
def get_embedding_function():
    """Create embeddings function with retry logic for rate limits"""
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        return embeddings
    except Exception as e:
        print(f"Error creating embedding function: {str(e)}")
        raise

def store_pdf_to_vectordb(pdf_path, collection_name):
    """
    Load a PDF file and store its contents in a Chroma vector database using Azure OpenAI
    with retry logic for rate limits
    """
    try:
        # 1. Load the PDF
        print("Loading PDF file...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # 2. Split the text into chunks
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(pages)
        print(f"Created {len(splits)} text chunks")
        
        # 3. Create embeddings
        print("Creating embedding function...")
        embedding_function = get_embedding_function()
        
        # 4. Create and store vectors in Chroma
        print("Storing vectors in database...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        
        # 5. Persist the database
        vectorstore.persist()
        print("Database successfully persisted")
        
        return vectorstore
        
    except Exception as e:
        print(f"Error in store_pdf_to_vectordb: {str(e)}")
        raise

def query_vectorstore(vectorstore, query_text, k=3):
    """Search the vector database for relevant content"""
    try:
        results = vectorstore.similarity_search(query_text, k=k)
        return results
    except Exception as e:
        print(f"Error in query_vectorstore: {str(e)}")
        raise

def main():
    # Print configuration for debugging
    print(f"Azure Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    print(f"API Key exists: {'Yes' if os.getenv('AZURE_OPENAI_API_KEY') else 'No'}")
    
    # Your specific PDF path
    pdf_path = "/Users/xrjin/Downloads/2401.05856v1.pdf"
    collection_name = "research_paper"
    
    try:
        # Store PDF in vector database
        print("Starting PDF processing pipeline...")
        vectorstore = store_pdf_to_vectordb(pdf_path, collection_name)
        print("PDF successfully stored in vector database")
        
        # Example query
        query = "What is the main topic of this research paper?"
        print("\nQuerying the database...")
        results = query_vectorstore(vectorstore, query)
        
        # Print results
        print("\nResults:")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print("Content:", doc.page_content)
            print("Metadata:", doc.metadata)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()