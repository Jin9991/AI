import os 
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes

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

# 1. Initialize the Azure OpenAI model
model = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
    deployment_name=deployment_name,
    api_version=api_version,
)

# 2. Create prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Please translate it to {language}"),
    ("user", "{text}") # Why ("user", "{text}") not work
])

# 3. Initialize the string parser
parser = StrOutputParser()

# 4. Create the chain
chain = prompt_template | model | parser

# 5. Use chain
try:
    result = chain.invoke({
        'language': 'Korean',
        'text': 'ChainLanguage is too hard to learn, but it worth it'
    })
    print("\nTranslation result:")
    print(result)
except Exception as e:
    print(f"An error occurred: {str(e)}")


# make our project as online ohter people can use 
# make fast api applicaiton

app = FastAPI(title = 'MyLangChainServe', version='V1.0', description='Use Langchain Translate any sentence serve')

add_routes(
    app,
    chain,
    path='/chainDemo',
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
