def generate_comparison_response(comparison_data, format_type='bullet'):
    """
    Generate a structured response for the comparison query.
    
    Args:
        comparison_data (List[str]): Aggregated comparison data.
        format_type (str): 'table' or 'bullet' for response format.
    
    Returns:
        str: Structured comparison response.
    """
    if format_type == 'table':
        response = "\n".join([f"Row {i+1}: {text}" for i, text in enumerate(comparison_data)])
        return f"Comparison Results:\n{response}"
    else:  # Default to bullet points
        response = "\n- ".join(comparison_data)
        return f"Comparison Results:\n- {response}"
