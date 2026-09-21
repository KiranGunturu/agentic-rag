import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv
from chunking import create_chunks

# Load the API key from the local .env file.
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Generate an embedding for the search text using the same model and dimension
# used when the documents were added to ChromaDB.
def get_embeddings(text):
        # Create one embedding vector for a single piece of text.
        response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=4,
        )
        return response.data[0].embedding

# Create embedding vectors for multiple chunks in one API request.
def get_embeddings_batch(texts):
        response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
                dimensions=4,
        )
        embeddings = []
        for item in response.data:
                embedding = item.embedding
                embeddings.append(embedding)
        return embeddings

# Open the persistent ChromaDB database and access the document collection.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection("hr_collection")

# Extract overlapping text chunks and their IDs and metadata from the PDF.
doc_path = "docs/HR.pdf"

chunks, ids, metadatas = create_chunks(doc_path, chunk_size=500, overlap_size=100)

# Generate vectors for every extracted chunk.
embeddings = get_embeddings_batch(chunks)

# Store the chunks, vectors, IDs, and metadata in the HR collection.
collection.add(ids = ids,
               documents=chunks,
               embeddings=embeddings,
               metadatas=metadatas)


