from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Initialize the vector store and embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("vectorstore_index", embeddings)

# 2. Define the user query
query = "Explain how to ensure factuality by incorporating retrieved data directly into the response."

# 3. Retrieve relevant context from the vector store
retrieved_contexts = vector_store.similarity_search(query, k=3)
retrieved_context = "\n".join([context.page_content for context in retrieved_contexts])

# 4. Define the prompt template
prompt_template = PromptTemplate(
    template="Based on the context provided: {context}, explain how to ensure factuality by incorporating the retrieved data directly into the response.",
    input_variables=["context"]
)

# 5. Initialize the language model
llm = OpenAI(model="gpt-4")

# 6. Chain the context and LLM to generate a response
chain = LLMChain(llm=llm, prompt=prompt_template)
response = chain.run(context=retrieved_context)

# 7. Output the response
print(response)
