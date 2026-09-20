from pypdf import PdfReader

doc_path = "C:/Users/kiran/OneDrive/Documents/AgenticRAG/docs/HR.pdf"


#print(f"Number of pages: {len(pdf_reader.pages)}")
#(pdf_reader.pages)

def create_chunks(text, chunk_size=500, overlap_size=100):
    pdf_reader = PdfReader(doc_path)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
        full_text += "\n"
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap_size
    return chunks

# chunks = create_chunks(doc_path, chunk_size=500, overlap_size=100)
# print(f"Number of chunks: {len(chunks)}")
