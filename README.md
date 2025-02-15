# Vector Embeddings

A vector database like **Chroma** is an open-source **vector store** used for storing and retrieving **vector embeddings**.

But what is a **vector embedding**? 

To understand what an **embedding** is, imagine you have a **thought** or a **sentence** in your mind. If I ask you to describe it to someone, you'd use **words**. But computers don't understand words the way we do, they only understand **numbers**.

An **embedding** is like a way of converting an idea (text or images) into a list of numbers, just like a **coordinate system** for meaning.

Think of a **huge multi-dimensional space**, not 2D or 3D, but maybe 768D or 1024D. Every sentence or word you input is placed somewhere in this space, based on its meaning. Similar sentences will have **similar coordinates**, meaning they are close together.

Since embeddings exist in a high-dimensional space, each value corresponds to a specific dimension of that space. For example the embedding `[0.56, -0.23, 0.89, ..., 0.12]`, we say `0.56` is the value of the first dimension of that embedding. You could also say `0.56` is the first component of the vector.

In order to use these coordinates, we have models like `sentence-transformers/all-mpnet-base-v2`, which is a **pretrained transformer**, meaning the model has been trained on a massive amount of text, learning how words relate to each other.

This adds **context awareness**, so the model understands that "*bank*" in "*river bank*" is different from "*bank*" in "*money bank*".

Instead of storing words as individual entities, it converts them into a set of **numbers (a vector)** that capture meaning, and this is called a **vector representation**.

For example, let's say we use an embedding model on these two sentences:

|Sentence                   |Embedding                       |
|:--------------------------|:------------------------------:|
|**"I love apples."**       |[0.017, ..., -0.019, ..., 0.034]|
|**"I enjoy eating fruit."**|[0.033, ..., -0.019, ..., 0.036]|

Even though they don't have the same words, their **embeddings** are pretty close together because they mean something similar.

These embeddings are then stored a in a **vector database** like **Chroma**, **Pinecone**  or **Weaviate**, which are designed to store and retrieve embeddings efficiently.

You can think of it like a library:
- Normally, libraries organize books by **titles or categories**.
- A vector database organizes text based on **meaning**.

### How it works

- When you embed text using the `embed_query(text)` function, **Chroma** stores that vector representation (see code below).
- When you query the database with a sentence, Chroma converts it to an embedding and finds the nearest vectors (other sentences that have similar meaning) and returns the most relevant ones.

Let's do a simple test using **LangChain**, an open source framework for building applications based on large language models (**LLM**s).

```bash
pip install -qU langchain-huggingface
```

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# This model converts text into numerical vectors that capture meaning.
# Think of it as mapping ideas into a high-dimensional space.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Each sentence is transformed into a vector that represents its meaning.
apple_vector = embeddings_model.embed_query("I love apples.")
fruit_vector = embeddings_model.embed_query("I enjoy eating fruit.")
```

If we run this and print the first 10 vector dimensions:

```ini
Dimension: 1
Apple vector: 0.017457755282521248
Fruit vector: 0.03377978131175041

Dimension: 2
Apple vector: 0.018669012933969498
Fruit vector: -0.03601612523198128

Dimension: 3
Apple vector: 0.03515832871198654
Fruit vector: 0.016169007867574692

Dimension: 4
Apple vector: 0.032574523240327835
Fruit vector: 0.0906355082988739

Dimension: 5
Apple vector: -0.019644882529973984
Fruit vector: -0.019928324967622757

Dimension: 6
Apple vector: -0.0270682405680418
Fruit vector: 0.029443178325891495

Dimension: 7
Apple vector: 0.08928196877241135
Fruit vector: 0.1352580487728119

Dimension: 8
Apple vector: -0.006618626415729523
Fruit vector: -0.04667004942893982

Dimension: 9
Apple vector: 0.037400227040052414
Fruit vector: 0.017919624224305153

Dimension: 10
Apple vector: 0.03484780713915825
Fruit vector: 0.03668517619371414
```

### Example 

1. **Store:** "The sky is blue." 
2. **Search:** "What color is the sky?"
3. **Chroma will find** "The sky is blue." because their vectors are close.



```python
import chromadb

# Connect to ChromaDB, our vector database, where we store and search embeddings.
# We're using persistent storage so data is saved between runs.
vector_db = chromadb.PersistentClient(path="./chroma_storage_test")

# A collection is like a table in a traditional database—it holds related data.
collection = vector_db.get_or_create_collection(name="test_collection")

# These sentences will be stored in our vector database for future searches.
documents = [
    "LangChain is the best framework!", # A fact? An opinion?
    "The sky is blue.",            # A simple, objective statement.
    "Fedora is just awesome."      # Biased? Maybe, but it's in our dataset!
]

# Each document is transformed into a vector that represents its meaning.
document_vectors = embedding_model.embed_documents(documents)

# Unique IDs for each document
document_ids = ["1", "2", "3"]

# Store documents and their embeddings in ChromaDB
test_collection.upsert(
    documents=documents,
    embeddings=document_vectors,
    ids=document_ids
)

# Convert the query into an embedding
query_vector = embedding_model.embed_query("What color is the sky?")

# Retrieve the collection (ensuring we're querying the right dataset)
collection = vector_db.get_collection("test_collection")

# Search for the closest match using vector similarity
results = collection.query(
    query_embeddings=[query_vector],
    n_results=1,  # We only want the single best match
    include=["documents"]  # Return the matching document, not just the ID
)

# Since our database understands meaning, it should return "The sky is blue."
print("Search result:", search_results['documents'][0][0])
```

This is called **vector search**, and it's what powers AI chatbots, recommendation systems and semantic search.

So, instead of exact **keyword matches**, vector databases allow for **semantic** matches, which means it understands **synonyms** and context. It retrieves information based on **meaning**, not just words and it's **faster** for AI applications, because looking up nearby vectors is much more efficient than scanning millions of words.

We learned that **embeddings** convert text into high-dimensional vectors that capture meaning.

Models like `all-MiniLM-L6-v2` and `all-mpnet-base-v2` generate these embeddings by understanding the relationships between words.

Vector databases like **Chroma** store and search these embeddings to find similar content.

At its core, embeddings let us **map meaning to numbers**, and vector databases let us **search by meaning, not just by words**.