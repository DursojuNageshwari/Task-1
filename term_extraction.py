import re

def extract_comparison_terms(query, comparison_keywords=["compare", "vs", "difference", "between"]):
    """
    Extract relevant terms for comparison from the user query.
    
    Args:
        query (str): User's natural language comparison query.
        comparison_keywords (list): Keywords indicating a comparison query.
    
    Returns:
        list: Extracted terms relevant to the comparison.
    """
    terms = []
    for keyword in comparison_keywords:
        if keyword in query.lower():
            match = re.search(f'({keyword})\\s+(.*?)\\s*(\\b(\\w+)\\b)', query.lower())
            if match:
                terms.extend(match.groups()[1:4])
    
    terms = list(set(terms))  # Remove duplicates
    return terms
