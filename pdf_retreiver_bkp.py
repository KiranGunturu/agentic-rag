from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb

# Load the API key and other environment variables from .env.
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(text):
        # Create one embedding vector for a single piece of text.
        response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=4,
        )
        return response.data[0].embedding


# Open the persistent ChromaDB database and access the document collection.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection("hr_collection")

# print(collection.count())

# print(collection.get(ids=['docs/HR.pdf_2']))

# Natural-language question used to search the HR document chunks.
query = "what is the leave policy"

# Convert the question into an embedding for similarity search.
query_embedding = get_embeddings(query)

# Retrieve the three chunks most relevant to the question.
related_docs = collection.query(query_embeddings=query_embedding, n_results=3)

#print(related_docs)

# print(related_docs['documents'])

# Combine the retrieved documents into the context for the language model.
context = related_docs['documents'][0]

# Build a prompt that asks the model to answer using only the retrieved context.
prompt = f"""

based on the given context answer user question

context: {context}

user_question: {query}

Rules: 
1. Give direct and natural answers
2. if you dont have a context for question, dont make up the answer just say I dont know

"""

# Generate a direct answer from the retrieved context.

respone = client.responses.create(model='gpt-5.6-sol',
                                  input = prompt)

# Display the generated answer.
print(respone.output_text)
