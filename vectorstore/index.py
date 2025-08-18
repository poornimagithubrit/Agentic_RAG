import os
import pandas as pd
import chromadb
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="vectorstore/")

def build_index(df: pd.DataFrame):
    collection = chroma_client.get_or_create_collection("csv_data")
    for i, row in df.iterrows():
        text = " | ".join([str(x) for x in row.values])
        embedding = client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[text])
    return collection
