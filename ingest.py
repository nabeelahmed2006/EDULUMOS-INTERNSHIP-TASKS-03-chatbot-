from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 1. Load the Encyclopedia
print("--- Step 1: Loading your Medical PDF ---")
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} pages from your PDF.")

# 2. Split into Chunks
print("--- Step 2: Splitting text into chunks ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. Create the 'Translator' (Embeddings)
print("--- Step 3: Downloading AI Model (This takes a moment) ---")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 4. Create the Vector Database
print("--- Step 4: Building the 'Brain' (Vectorstore) ---")
# This creates a folder named 'vectorstore' where the knowledge is saved
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings, 
    persist_directory="./vectorstore"
)

print("\nSUCCESS! Your bot has finished reading the Gale Encyclopedia.")
print("You should now see a folder named 'vectorstore' in your project.")