# ingest.py
import os
import glob
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# -------- 設定 --------
CHROMA_DIR = "chroma_db"         # 永続化先
COLLECTION = "rag_docs"          # コレクション名
MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 500                 # 文字ベース分割（単純）
CHUNK_OVERLAP = 50

# -------- ユーティリティ --------
def split_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    i, n = 0, len(text)
    step = max(1, size - overlap)
    while i < n:
        chunks.append(text[i:i+size])
        i += step
    return chunks

def read_txt(path: str) -> str:
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def read_md(path: str) -> str:
    # mdもテキストとして扱う（必要ならfrontmatter除去など拡張）
    return read_txt(path)

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def load_file(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path), "text"
    if ext == ".md":
        return read_md(path), "markdown"
    if ext == ".pdf":
        return read_pdf(path), "pdf"
    return "", "unknown"

# -------- メイン処理 --------
def main():
    # 1) ベクトルDB & 埋め込みモデル
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    embed = SentenceTransformer(MODEL_NAME)

    files = glob.glob("docs/**/*.txt", recursive=True) \
          + glob.glob("docs/**/*.md", recursive=True) \
          + glob.glob("docs/**/*.pdf", recursive=True)

    if not files:
        print("docs/ に対象ファイルが見つかりません。")
        return

    id_batch, doc_batch, meta_batch, emb_batch = [], [], [], []

    for path in files:
        text, kind = load_file(path)
        if not text.strip():
            continue

        # 2) 分割
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        # 3) e5は prefix 推奨
        #   - passage: <chunk> で埋め込む
        embs = embed.encode([f"passage: {c}" for c in chunks],
                            normalize_embeddings=True).tolist()

        # 4) 追加バッチ作成
        for i, (c, e) in enumerate(zip(chunks, embs)):
            id_batch.append(f"{path}:{i}")
            doc_batch.append(c)
            meta_batch.append({"source": path, "type": kind})
            emb_batch.append(e)

    if not id_batch:
        print("登録対象のチャンクがありません。")
        return

    # 5) upsert（存在すれば置換）
    # Chromaは add が基本なので、簡易に一度 delete→add でもOK
    # 同一IDを更新したい場合は一旦削除:
    try:
        col.delete(ids=id_batch)
    except Exception:
        pass

    col.add(ids=id_batch, documents=doc_batch, metadatas=meta_batch, embeddings=emb_batch)
    print(f"登録完了: {len(id_batch)} chunks")

if __name__ == "__main__":
    main()
