import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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

# Creating custome retreiver compatible with LCEL
class CustomPineconeRetriever:
    def __init__(self, index, namespace):
        self.index = index
        self.namespace = namespace

    def invoke(self, query, top_k=3):
        retrieved_data = self.index.search(
            namespace=self.namespace,
            query={"inputs": {"text": query}, "top_k": top_k}
        )
        return [hit.fields['text'] for hit in retrieved_data.result.hits]
    

def retrieve_and_answer_with_lcel(question):
    """
    Retrieval chain using LCEL with a custom retriever.
    The retriever is defined as a class that implements the invoke method, making it compatible with LCEL.

    Benefits:
    - Cleaner and more modular code
    - Built-in support for streaming and async operations
    - Easier to compose with other chains
    - Less error-prone and more maintainable
    
    """
    
    retriever = CustomPineconeRetriever(index=index, namespace=os.environ["NAMESPACE"])
    
    retiever_runnable = RunnableLambda(retriever.invoke)
    
    llm_chain = ({"context": retiever_runnable | format_docs,
                 "question": RunnablePassthrough()} 
                 | prompt_template 
                 | llm)
    
    llm_chain_output = llm_chain.invoke(question)
    
    return llm_chain_output


if __name__ == "__main__":
    question = "What is Pinecone in Machine Learning?"
    answer = retrieve_and_answer_with_lcel(question)
    print("Answer:", answer.content)