from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb

# Use a persistent database so the collection remains available between runs.

# Load environment variables, including the OpenAI API key.
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create an embedding vector for a piece of text using OpenAI.
def get_embeddings(text):
    response = client.embeddings.create(model="text-embedding-3-small", 
                            input=text,
                            dimensions=4)
    return response.data[0].embedding

documents = ["spark is a big data processing framework.",
            "aws is a cloud computing service.",
            "data science is a field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
            "machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "deep learning is a subset of machine learning that uses neural networks with many layers to learn from data."
            ]

# Generate embeddings for all source documents before storing them.
doc_embeddings = []
for text in documents:
    doc_embeddings.append(get_embeddings(text))

# Open the persistent ChromaDB database and select the document collection.
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("chromadb_collection")

# Store the documents, their IDs, and their corresponding embedding vectors.
collection.add(
    documents=documents,
    ids=["id1", "id2", "id3", "id4", "id5"],
    embeddings=doc_embeddings
)

