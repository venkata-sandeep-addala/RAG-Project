from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

def ingestion_pipeline():
    # Load the document
    print("Loading document...")
    loader = TextLoader("D:\Projects\RAG-Project\mediumblog1.txt", encoding="utf-8")
    documents = loader.load()
    
    # Split the document into chunks
    print("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    
    # Ingesting documents
    print("Ingesting documents into Pinecone...")
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    
    pc = Pinecone()
    index = pc.Index(index_name="rag-personal-project-1", host="https://rag-personal-project-1-qiplvg3.svc.aped-4627-b74a.pinecone.io")

    index.upsert_records(
        namespace="default",
        records=[
            {
                "id": f"id-{i}",
                "text": texts[i], 
                "metadata": str(metadatas[i].get("source", ""))
            }
            for i in range(len(texts))
        ]
    )
    
    
    
    print(f"✅ Ingested {len(chunks)} chunks")


if __name__ == "__main__":
    ingestion_pipeline()
