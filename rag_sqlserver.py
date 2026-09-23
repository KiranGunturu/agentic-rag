import os
from dotenv import load_dotenv
import pyodbc
from openai import OpenAI

from chunking import create_chunks

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
DB_SERVER = "localhost\\MSSQLSERVER03"
DB_NAME = "retail"
TABLE_NAME = "dbo.hr_policy_docs"


def get_embeddings_batch(texts):
    """Create embedding vectors for multiple chunks in one API request."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


def get_sql_connection():
    """Create a connection using Windows Authentication for the named local SQL Server instance."""
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)


# Create chunks from the HR PDF and keep the returned tuple structure consistent.
chunks, ids, metadatas = create_chunks("docs/HR.pdf", 500, 100)

# Filter out blank chunks so the embedding API receives only real strings.
valid_chunks = [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
valid_ids = [ids[idx] for idx, chunk in enumerate(chunks) if isinstance(chunk, str) and chunk.strip()]
valid_metadatas = [metadatas[idx] for idx, chunk in enumerate(chunks) if isinstance(chunk, str) and chunk.strip()]

# Generate embeddings matching the SQL Server VECTOR(1536) column schema.
embeddings = get_embeddings_batch(valid_chunks)

# Insert each chunk and its embedding into the local SQL Server table.
# SQL Server rejects binding the vector as a parameter, so embed the vector as a
# literal inside the SQL string while keeping the text content parameterized.
conn = get_sql_connection()
cursor = conn.cursor()

for content, embedding in zip(valid_chunks, embeddings):
    vector_literal = "[" + ",".join(str(float(value)) for value in embedding) + "]"
    safe_content = content.replace("'", "''")
    cursor.execute(
        f"""
        INSERT INTO {TABLE_NAME} (content, embedding)
        VALUES ('{safe_content}', CAST('{vector_literal}' AS VECTOR({EMBEDDING_DIMENSIONS})))
        """
    )

conn.commit()
print(f"Inserted {len(valid_chunks)} chunks into {TABLE_NAME} in database {DB_NAME}.")

cursor.close()
conn.close()

