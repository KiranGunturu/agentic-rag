from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read a PDF and recursively split the extracted text into smaller, overlapping chunks.
def create_chunks(pdf_path, chunk_size=500, overlap_size=100):
    # Validate the chunking parameters before splitting.
    if chunk_size <= 0 or overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("chunk_size must be positive and greater than overlap_size")

    # Open the PDF and combine all pages into a single string.
    pdf_reader = PdfReader(pdf_path)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() or ""
        full_text += "\n"

    # Use a recursive text splitter so the text is broken into smaller chunks with overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(full_text)
    return texts

# Example usage: create chunks for the HR PDF and print a few sample chunks.
chunks = create_chunks('docs/HR.pdf')
print(f"Number of chunks: {len(chunks)}")
print(chunks[0], '\n')
print(chunks[1], '\n')
print(chunks[2], '\n')
