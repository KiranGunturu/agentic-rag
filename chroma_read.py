import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the API key from the local .env file.
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Generate an embedding for the search text using the same model and dimension
# used when the documents were added to ChromaDB.
def get_embeddings(text):
        response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=4,
        )
        return response.data[0].embedding

# Open the persistent ChromaDB database and access the document collection.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection("chromadb_collection")

# Convert the natural-language query into an embedding vector.
query = "aws is a cloud computing service."  # Text used to find similar documents.

question_embedding = get_embeddings(query)

# Return the five documents most similar to the query embedding.
# Query ChromaDB with the generated vector instead of the raw text.
records = collection.query(query_embeddings=[question_embedding], n_results=5)

print(records)



