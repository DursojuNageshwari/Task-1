from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load pre-trained model
    embeddings = model.encode(chunks)
    return embeddings

# Example chunks
chunks = [
    "This is the first chunk of text.",
    "Here is another chunk of text with some additional information.",
    "The third chunk contains even more detailed content."
]

# Generate embeddings for the chunks
embeddings = generate_embeddings(chunks)
print(f"Generated {len(embeddings)} embeddings")
