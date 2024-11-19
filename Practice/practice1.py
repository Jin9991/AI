import os


from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
embeddings = AzureOpenAIEmbeddings(model=os.getenv("AZURE_EMBEDDING_DEPLOY_NAME"))
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

retriever = db.as_retriever()
print(retriever)


loader = TextLoader("./就業規則.md", encoding='utf8')
documents = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=80,
    chunk_overlap=0,
)

docs = text_splitter.split_documents(documents)
print(docs)
print(len(docs))
