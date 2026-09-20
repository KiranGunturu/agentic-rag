import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("chromadb_collection")

documents = ["spark is a big data processing framework.",
            "aws is a cloud computing service.",
            "data science is a field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
            "machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "deep learning is a subset of machine learning that uses neural networks with many layers to learn from data."
            ]

collection.add(
    documents=documents,
    ids=["id1", "id2", "id3", "id4", "id5"]
)

