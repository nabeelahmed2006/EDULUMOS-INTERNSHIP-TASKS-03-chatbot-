# store_index.py

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from dotenv import load_dotenv
import os


# ------------------------
# Load Environment Variables
# ------------------------

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


# ------------------------
# Load and Process Data
# ------------------------
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# ------------------------
# Initialize Pinecone
# ------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# Create index only if it doesn't exist
existing_indexes = [index["name"] for index in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully ✅")
else:
    print("Index already exists ✅")


# ------------------------
# Connect to Index
# ------------------------

index = pc.Index(index_name)

vector_store = LC_Pinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)


# ------------------------
# Upload Documents
# ------------------------

vector_store.add_documents(text_chunks)

print("Documents uploaded successfully ✅")