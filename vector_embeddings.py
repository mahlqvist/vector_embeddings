from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import chromadb

# Load an embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to ChromaDB
vector_db = chromadb.PersistentClient(path="./chroma_storage_test")

# Create (or get) a collection
test_collection = vector_db.get_or_create_collection(name="test_collection")

# Define some documents
documents = [
    "LangChain is the best framework!",  
    "The sky is blue.",                  # A simple, objective statement.
    "Fedora is just awesome."             
]

# Convert documents into embeddings
document_vectors = embedding_model.embed_documents(documents)

# Unique IDs for each document
document_ids = ["1", "2", "3"]  

# Store documents and their embeddings in ChromaDB
test_collection.upsert(
    documents=documents,
    embeddings=document_vectors,
    ids=document_ids
)

# Define a search function
def query_chroma_db(query_text):
    """
    Given a text query, find the most semantically similar document in ChromaDB.
    """
    # Convert the query into an embedding
    query_vector = embedding_model.embed_query(query_text)

    # Retrieve the collection (ensuring we're querying the right dataset)
    collection = vector_db.get_collection("test_collection")
    
    # Search for the closest match using vector similarity
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1,  # We only want the single best match
        include=["documents"]  # Return the matching document, not just the ID
    )

    return results

# Test the search function
search_results = query_chroma_db("What color is the sky?")

# Print the best match
# Since our database understands meaning, it should return "The sky is blue."
print("Search result:", search_results['documents'][0][0])