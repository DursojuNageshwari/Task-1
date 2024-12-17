def aggregate_comparison_results(results):
    """
    Aggregate and process the retrieved chunks for comparison.
    
    Args:
        results (List[dict]): Retrieved chunks from the vector database.
    
    Returns:
        List[str]: Processed and aggregated comparison results.
    """
    aggregated_results = []
    
    for chunk in results['matches']:
        text = chunk['text']  # Extract the relevant chunk's text
        similarity_score = chunk['score']  # Optional: Use similarity scores if needed for ranking.
        
        aggregated_results.append(text)
    
    return aggregated_results
