from sentence_transformers import SentenceTransformer

# Function to generate query embedding
def get_query_embedding(query, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)  # Load pre-trained model
    query_embedding = model.encode([query], convert_to_tensor=True)  # Generate query embedding
    return query_embedding

# Example query
query = "What are the key trends in the 2015 U.S. GDP data?"
query_embedding = get_query_embedding(query)
print(query_embedding)
