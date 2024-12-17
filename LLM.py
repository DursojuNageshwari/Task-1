import openai

# Function to generate a response using LLM with retrieval-augmented context
def generate_response_with_context(retrieved_chunks, user_query, model="gpt-4", temperature=0.2):
    """
    Generate a response using an LLM with retrieval-augmented prompts.
    
    Parameters:
    - retrieved_chunks (list): List of text chunks retrieved from the vector database.
    - user_query (str): The user's query.
    - model (str): LLM model to use (default: gpt-4).
    - temperature (float): LLM creativity parameter (default: 0.2 for factuality).
    
    Returns:
    - str: The generated response.
    """
    
    # Step 1: Combine the retrieved chunks into a structured context
    context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    # Step 2: Construct the retrieval-augmented prompt
    prompt = f"""
    You are tasked with answering the user's question based on the provided context.  
    Ensure that your response is factual, concise, and uses the retrieved data directly.  

    Context:  
    {context}

    User Query:  
    {user_query}

    Response:
    """
    
    # Step 3: Call the LLM API to generate the response
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides factual answers based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        
        # Step 4: Extract and return the LLM's response
        generated_response = response['choices'][0]['message']['content'].strip()
        return generated_response
    
    except Exception as e:
        return f"An error occurred: {e}"

# Example Usage
if _name_ == "_main_":
    # Example retrieved chunks from a vector database
    retrieved_chunks = [
        "Document X: Refunds are allowed within 30 days for unused items.",
        "Document Y: Refunds are allowed within 15 days for items in original packaging."
    ]
    
    # User query
    user_query = "What are the refund policies for Document X and Document Y?"
    
    # Generate the response
    response = generate_response_with_context(retrieved_chunks, user_query)
    
    # Print the response
    print("Generated Response:")
    print(response)