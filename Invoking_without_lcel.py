import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()


# Initialize Pinecone client
pc = Pinecone()

index = pc.Index(os.environ["INDEX_NAME"])

# Initialize ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")

def format_docs(docs):
    """Format the retrieved documents for better readability."""
    return '\n\n'.join(docs)

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based on the retrieved context:

{context}

Question: {question}

"""
)

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
    retrieved_data = index.search(
        namespace = os.environ["NAMESPACE"],
        query = {"inputs":{"text": question}, "top_k": 3}
    )
    
    # Format the retrieved documents
    context = format_docs([hit.fields['text'] for hit in retrieved_data.result.hits])
    
    # Generate an answer using ChatGroq
    prompt = prompt_template.format(context=context, question=question)
    answer = llm.invoke(prompt)
    
    return answer.content


if __name__ == "__main__":
    question = "What is Pinecone in Machine Learning?"
    answer = retrieve_and_answer_without_lcel(question)
    print("Answer:", answer)