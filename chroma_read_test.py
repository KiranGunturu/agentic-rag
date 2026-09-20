import chromadb

# Connect to the database and open the existing document collection.

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("chromadb_collection")

#records = collection.get(ids=["id1"])

#records = collection.get(ids=["id1"], include=["metadatas", "documents", "embeddings"])

# Find the three most similar documents and include their metadata, text, and vectors.
records = collection.query(
    query_texts=["aws is a cloud computing service."], n_results=3, include=["metadatas", "documents", "embeddings"])

print(records)

# The embedding model and vector dimensions must match during storage and retrieval.

