# from huggingface_hub import snapshot_download

# model_path = snapshot_download(
#     repo_id="BAAI/bge-reranker-v2-m3",
#     local_dir="./BAAI/bge-reranker-v2-m3"
# )

# print(model_path)

from dotenv import load_dotenv
import os
load_dotenv()
import numpy
from sentence_transformers import CrossEncoder

# A cross-encoder reads the question and chunk together to score their relevance.
# Unlike a bi-encoder, it does not embed them independently and compare vectors.
model = CrossEncoder(model_name_or_path = 'BAAI/bge-reranker-v2-m3')

def reranker(query, docs, top_k):
    # Pair the same question with each chunk so every pair is scored jointly.
    pairs = [
        [query, document]
        for document in docs
    
    ]

    scores = model.predict(pairs)

    # Keep documents ordered from the highest relevance score to the lowest.
    ranked_documents = sorted(
        zip(docs, scores),
        key=lambda x:x[1],
        reverse=True
    )

    # Return only the requested number of top-ranked documents.
    top_documents = [
        document
        for document, score in ranked_documents[:top_k]
    ]

    return top_documents


# Example candidate passages used to demonstrate reranking.
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


query = "how many leaves i can carry forward to next year"

# Print the three passages most relevant to the example query.
top_documents = reranker(
    query, documents, 3
)

print(top_documents)