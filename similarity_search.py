def search_similar_chunks(query_embedding, collection, n_results=3):
    """
    Search for similar chunks in the vector database.
    
    Args:
        query_embedding (numpy.ndarray): Vector embedding of the user's query.
        collection: The vector database collection instance (e.g., ChromaDB).
        n_results (int): Number of top chunks to retrieve.
    
    Returns:
        dict: Retrieved chunks, including their text and metadata.
    """
    results = collection.query(
        query_embeddings=[query_embedding],  # Query vector
        n_results=n_results  # Number of top similar chunks
    )
    return results
