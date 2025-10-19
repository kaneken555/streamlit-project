# verify_chroma.py
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("rag_docs")

print("[COUNT]", col.count())

# すべてのメタデータを確認
items = col.get(include=["metadatas", "documents"], limit=50)
for meta in items["metadatas"]:
    print(meta["source"], meta.get("date"))
