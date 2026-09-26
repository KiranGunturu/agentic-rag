from rank_bm25 import BM25Okapi
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess(doc):
    # Normalize text and split it into tokens for keyword-based scoring.
    tokenized_doc = doc.lower()
    tokenized_doc = re.sub(r"[^\w\s]","", tokenized_doc)
    tokenized_doc = tokenized_doc.split(" ")
    return [stemmer.stem(token) for token in tokenized_doc]

# Example policy passages used as the BM25 search corpus.
documents = [
    "we get 10 annual leave. leave cannot be carry forwarded",
    "Employees can carry forward a maximum of 6 unused annual leave days to the next calendar year.",
    "Employees receive 18 days of annual leave every year.",
    "Employees can take up to 6 days of sick leave for medical reasons.",
    "Unused leave cannot be carried forward to the next year.",
    "Employees may carry forward unused compensatory off days, subject to manager approval.",
    "Annual leave requests must be submitted at least one week in advance.",
    "Employees can accumulate up to 30 days of annual leave over multiple years."
    
]

query = "how many leaves i can carry forward to next year"

# Apply the same preprocessing to the query and every document.
query_tokenize = preprocess(query)
docs_tokenize = []

for doc in documents:
    docs_tokenize.append(preprocess(doc))

# print(docs_tokenize)

# Build BM25's corpus statistics for later keyword-relevance scoring.
bm25 = BM25Okapi(docs_tokenize)

# Print the statistics calculated while initializing the BM25 index.
# print("\nDocument lengths:")
# print(bm25.doc_len)

# print("\nDocument freqencies:")
# print(bm25.doc_freqs)

# print("\nIDF:")
# print(bm25.idf)

# print("\nAvg document length:")
# print(bm25.avgdl)


scores = bm25.get_scores(query_tokenize)
print(scores)