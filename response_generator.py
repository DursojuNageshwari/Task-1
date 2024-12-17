import openai

def generate_response_with_context(retrieved_chunks, query, model="gpt-4"):
    """
    Generate a response using an LLM with the retrieved chunks as context.
    
    Args:
        retrieved_chunks (list): List of text chunks retrieved from the vector database.
        query (str): User's question.
        model (str): LLM model to use (default: 'gpt-4').
    
    Returns:
        str: Generated response from the LLM.
    """
    # Combine retrieved chunks into a single context
    context = "\n\n".join(retrieved_chunks)
    
    # Construct the prompt
    prompt = f"""
    You are an AI assistant tasked with answering questions based on the provided context.

    Context:
    {context}

    Question: {query}

    Provide a detailed and accurate response based on the above context.
    """
    # Call the LLM (OpenAI GPT-4 in this case)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
