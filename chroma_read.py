import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("chromadb_collection")

#records = collection.get(ids=["id1"])

#records = collection.get(ids=["id1"], include=["metadatas", "documents", "embeddings"])

# give me a query to find the most similar document to "aws is a cloud computing service." and return the top 3 results with their metadata, documents, and embeddings.
records = collection.query(
    query_texts=["aws is a cloud computing service."], n_results=3, include=["metadatas", "documents", "embeddings"])

print(records)

