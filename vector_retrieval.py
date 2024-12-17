from typing import List
import numpy as np

def retrieve_comparison_chunks(terms, collection, n_results=3):
    """
    Retrieve chunks from the vector database relevant to comparison terms.
    
    Args:
        terms (List[str]): Terms extracted from the user's query.
        collection: The vector database collection (e.g., ChromaDB).
        n_results (int): Number of top results to retrieve.
    
    Returns:
        List[dict]: Retrieved chunks, including text and metadata.
    """
    query_embeddings = [collection.embed(term) for term in terms]
    results = collection.query(
        query_embeddings=query_embeddings,  # List of term vectors
        n_results=n_results
    )
    return results

