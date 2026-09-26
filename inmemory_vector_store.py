from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
load_dotenv()
from reranking import reranker

# Sample text chunks to index and search semantically.
documents = [
    "Employees can carry forward a maximum of 6 unused annual leave days to the next calendar year.",
    "Employees receive 18 days of annual leave every year.",
    "Employees can take up to 6 days of sick leave for medical reasons.",
    "Unused leave cannot be carried forward to the next year.",
    "Employees may carry forward unused compensatory off days, subject to manager approval.",
    "Annual leave requests must be submitted at least one week in advance.",
    "Employees can accumulate up to 30 days of annual leave over multiple years.",
    "Unused annual leave exceeding the carry-forward limit will expire at the end of the year.",
    "Employees receive 12 days of sick leave every year.",
    "Public holidays are separate from annual leave and do not count toward the annual leave balance.",
    "Employees can carry forward up to 3 unused casual leave days.",
    "Managers must approve all leave requests through the leave management system.",
    "Employees get 10 casual leaves every year.",
    "Employees are entitled to a 1-hour lunch break from 1:00 PM to 2:00 PM during normal working days.",
    "Employees absent for more than 5 consecutive days without informing the company may face termination."
]

# Convert text and search queries into vectors with the same embedding model.
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# Create a temporary vector store that keeps its index in memory.
vector_store = InMemoryVectorStore(embedding)

# Embed and add each document chunk to the vector store.
vector_store.add_texts(texts=documents)

query = "how many leaves i can carry forward to next year"

# Retrieve the five chunks with vectors most similar to the query.
results = vector_store.similarity_search(query,
                                         k=10)

# print(results)

retreived_docs = []
for doc in results:
    retreived_docs.append(doc.page_content)

# print(retreived_docs)

reranked_top3 = reranker(query, retreived_docs, 3)
print(reranked_top3)