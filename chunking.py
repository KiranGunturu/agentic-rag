from pypdf import PdfReader

# PDF source used for text extraction.

#print(f"Number of pages: {len(pdf_reader.pages)}")
#(pdf_reader.pages)

def create_chunks(pdf_path, chunk_size=500, overlap_size=100):
    # Read the PDF and split its text into overlapping chunks.
    if chunk_size <= 0 or overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("chunk_size must be positive and greater than overlap_size")

    pdf_reader = PdfReader(pdf_path)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() or ""
        full_text += "\n"
    
    # Overlap preserves context between neighboring chunks.
    ids = []
    chunks = []
    metadatas = []
    start = 0
    chunk_id = 1
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        ids.append(f'{pdf_path}_{chunk_id}')
        metadatas.append({"doc_name": pdf_path, "version": 1})
        chunk_id = chunk_id + 1
        start = end - overlap_size
    return chunks, ids, metadatas

chunks, ids, metadatas = create_chunks('docs/HR.pdf', chunk_size=500, overlap_size=100)
#print(f"Number of chunks: {len(chunks)}")
# print(chunks[0])
# print(ids[0])
# print(metadatas[0])
