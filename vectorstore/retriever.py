import os
import chromadb
import pandas as pd
from openai import OpenAI

client = OpenAI(OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"))


# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="vectorstore/")
collection = chroma_client.get_or_create_collection("csv_data")

def embed_text(text: str):
    """Generate embedding using OpenAI API"""
    embedding = client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding
    return embedding

def index_csv(df: pd.DataFrame):
    """Index all rows from DataFrame into Chroma vector DB"""
    collection.delete(where={})  # clear old index
    for i, row in df.iterrows():
        text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
        embedding = embed_text(text)
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[text])
    return {"status": "indexed", "rows": len(df)}

def retrieve(query: str, top_k: int = 5):
    """Retrieve top-k relevant rows for a query"""
    embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "distances", "ids"]
    )

    matches = []
    for doc, dist, idx in zip(results["documents"][0], results["distances"][0], results["ids"][0]):
        matches.append({"id": idx, "text": doc, "score": 1 - dist})  # similarity score
    return matches
