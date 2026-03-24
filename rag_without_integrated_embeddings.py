import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

pc = Pinecone()
index = pc.Index(name=os.environ["INDEX_NAME2"])

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})



# Initialize Groq model
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based on the retrieved context:

{context}

Question: {question}

""")


def retrieve_and_answer_without_lcel(question):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    
    """
    
    
    # Retrieve relevant documents from Pinecone
    query_vector = embedding_model.embed_query("What is Pinecone?")

    retrieved_docs = index.query(vector=query_vector, top_k=3, include_metadata=True)
    
    if not retrieved_docs:
        return "No relevant documents found."
    
    # Format the retrieved documents
    context = '\n\n'.join([match['metadata']['text'] for match in retrieved_docs.matches])
    
    # Generate an answer using ChatGroq
    prompt = prompt_template.format(context=context, question=question)
    answer = llm.invoke(prompt)
    
    return answer.content

if __name__ == "__main__":
    question = "What is Pinecone in Machine Learning?"
    response = retrieve_and_answer_without_lcel(question)
    # answer = embedding_model.embed_query(question)
    print("Answer:", response)