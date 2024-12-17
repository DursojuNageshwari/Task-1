import faiss
import numpy as np

def store_embeddings(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)  # L2 similarity
    index.add(np.array(embeddings))  # Add embeddings
    return index

# Example usage
# Suppose 'embeddings' is the list of embeddings generated earlier
index = store_embeddings(embeddings)
print(f"Index has {index.ntotal} vectors")
