def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Example usage
pdf_text = "This is an example text extracted from a PDF file. " * 50  # Simulating extracted text
chunks = chunk_text(pdf_text)
print(f"Total chunks: {len(chunks)}")
print(chunks[0])  # Display the first chunk
