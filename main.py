from openai import OpenAI
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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

doc_embeddings = []
for text in documents:
    doc_embeddings.append(get_embeddings(text))

#print(doc_embeddings)

query = "aws is a cloud computing service."
query_embedding = get_embeddings(query)
#print(query_embedding)

similarities = cosine_similarity([query_embedding], doc_embeddings)
print(similarities)

# print(len(response.data[0].embedding))
# print(response.data[0].embedding)