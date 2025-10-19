# ingest.py
import os
import re
import glob
import hashlib
from typing import List, Tuple, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# -------- 設定 --------
CHROMA_DIR = "chroma_db"          # 永続化先
COLLECTION_NAME = "rag_docs"      # コレクション名
MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 500                  # 文字ベース（まずは簡易）
CHUNK_OVERLAP = 50
BATCH_SIZE = 1000                 # Chromaへの追加バッチ
DOCS_DIRS = ["docs"]              # 追加で "notes", "papers" など増やせる

# -------- ユーティリティ --------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def simple_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def split_text(text: str, size: int, overlap: int) -> List[str]:
    """
    文字数ベースのシンプル分割。
    改善余地: 句読点/改行で先に軽く分割 → パックしてsizeに近づける。
    """
    chunks = []
    i, n = 0, len(text)
    step = max(1, size - overlap)
    while i < n:
        chunks.append(text[i:i + size])
        i += step
    return chunks

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_md(path: str) -> str:
    # 必要ならFrontMatter除去やコードブロック除去を追加
    return read_txt(path)

def read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        print(f"[WARN] PDF読取失敗: {path} ({e})")
        return ""

def load_file(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path), "text"
    if ext == ".md":
        return read_md(path), "markdown"
    if ext == ".pdf":
        return read_pdf(path), "pdf"
    return "", "unknown"

def iter_target_files() -> List[str]:
    patterns = []
    for base in DOCS_DIRS:
        patterns += glob.glob(os.path.join(base, "**", "*.txt"), recursive=True)
        patterns += glob.glob(os.path.join(base, "**", "*.md"), recursive=True)
        patterns += glob.glob(os.path.join(base, "**", "*.pdf"), recursive=True)
    # 安定した処理順（再現性）
    return sorted(set(patterns))

# -------- Markdownの日付抽出 --------
def extract_date_from_text(text: str) -> str:
    """
    Markdownの先頭付近から「日付: YYYY-MM-DD」を抽出
    - 柔軟に '日付：' や 'Date:' などにも対応
    """
    head = "\n".join(text.splitlines()[:30])  # 冒頭30行を対象に
    m = re.search(r"(?:日付|Date)\s*[:：]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", head)
    if m:
        return m.group(1)
    return ""

# ユーティリティ群の下あたりに追記（NEW）
def normalize_tags(tag_text: str) -> List[str]:
    """
    'タグ: [Streamlit, LLM, UI]' / 'タグ: Streamlit, LLM, UI'
    'タグ：Streamlit｜LLM｜UI' などを配列へ正規化
    """
    if not tag_text:
        return []
    s = tag_text.strip()
    # 角括弧で囲まれていれば外す
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # よくある区切り文字をカンマに寄せる（全角含む）
    for sep in ("｜", "|", "、", "，", ";"):
        s = s.replace(sep, ",")
    # 分割してトリム、空要素除去
    tags = [t.strip() for t in s.split(",") if t.strip()]
    # 重複排除（順序維持）
    seen, uniq = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def extract_tags_from_text(text: str) -> List[str]:
    """
    Markdownの冒頭30行から 'タグ:' 行を抽出して配列化
    例:
      タグ: [Streamlit, LLM, UI]
      タグ：Streamlit, LLM, UI
      Tags: Streamlit｜LLM｜UI
    """
    head = "\n".join(text.splitlines()[:30])
    m = re.search(r"^(?:タグ|Tags?)\s*[:：]\s*(.+?)\s*$", head, flags=re.MULTILINE | re.IGNORECASE)
    if not m:
        return []
    return normalize_tags(m.group(1))

# 日付・タグの抽出関数の近くに追記（NEW）
def parse_study_time_hours(line: str) -> float:
    """
    '学習時間: 4時間' / '学習時間: 1.5時間' / '学習時間: 90分' / '学習時間: 90 m' 等を時間(float)へ正規化
    単位省略（数値のみ）の場合は「時間」とみなす
    """
    s = (line or "").strip()

    # 時間（小数対応）
    m_h = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*時(?:間)?", s)
    if m_h:
        return float(m_h.group(1))

    # 分（整数）
    m_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*分|([0-9]+)\s*m(in)?", s, flags=re.IGNORECASE)
    if m_m:
        # グループのどちらかにマッチ
        val = m_m.group(1) or m_m.group(2)
        return round(float(val) / 60.0, 4)

    # 数値のみ（単位省略は時間と解釈）
    m_n = re.search(r"[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*$", s)
    if m_n:
        return float(m_n.group(1))

    return 0.0

def extract_study_time_from_text(text: str) -> float:
    """
    Markdown冒頭30行から '学習時間:' を抽出し、時間(float)にして返す
    """
    head = "\n".join((text or "").splitlines()[:30])
    m = re.search(r"^\s*学習時間\s*[:：]\s*(.+?)\s*$", head, flags=re.MULTILINE)
    return parse_study_time_hours(m.group(1)) if m else 0.0



# -------- メイン処理 --------
def main():
    # 0) 前提チェック
    ensure_dir(CHROMA_DIR)
    if not any(os.path.isdir(d) for d in DOCS_DIRS):
        print(f"[INFO] 対象ディレクトリがありません: {DOCS_DIRS}")
    files = iter_target_files()
    if not files:
        print("[INFO] 対象ファイルが見つかりません。")
        return

    # 一覧表示
    print(f"[SCAN] {len(files)} files")
    for p in files:
        print(" -", p)

    # 1) ベクトルDB & 埋め込みモデル
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    embed = SentenceTransformer(MODEL_NAME)  # , device="cuda"

    ids_buf, docs_buf, metas_buf, embs_buf = [], [], [], []

    # 2) 各ファイルを処理
    for path in files:
        abs_path = os.path.abspath(path)  # ← 絶対パスで統一（重要）
        text, kind = load_file(abs_path)
        print(f"[READ] {abs_path} len={len(text)} kind={kind}")

        if not text or not text.strip():
            print(f"[SKIP-EMPTY] {abs_path}")
            continue

        file_hash = simple_hash(text)
        date_str = extract_date_from_text(text)
        tags_list = extract_tags_from_text(text) 
        study_time_hours = extract_study_time_from_text(text)

        # 既存分を削除（source=絶対パスで一致させる）
        try:
            col.delete(where={"source": abs_path})
            print(f"[DEL] source={abs_path}")
        except Exception as e:
            print(f"[WARN] delete失敗 (source={abs_path}): {e}")

        # 3) チャンク化
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"[CHUNK] {abs_path} -> {len(chunks)} chunks")

        if not chunks:
            print(f"[SKIP-NOCHUNK] {abs_path}")
            continue

        # 4) E5 prefix（passage: ...）
        embs = embed.encode(
            [f"passage: {c}" for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        # 5) 追加キューに積む
        start_idx = len(ids_buf)
        for i, (c, e) in enumerate(zip(chunks, embs)):
            ids_buf.append(f"{abs_path}:{i}")
            docs_buf.append(c)
            metas_buf.append({
                "source": abs_path,
                "type": kind,
                "chunk_index": i,
                "file_hash": file_hash,
                "date": date_str,  # ← ✅ 日付を追加
                "tags_csv": ",".join(tags_list) if tags_list else None, 
                "study_time_hours": study_time_hours,  # ← ✅ 学習時間を追加
            })
            embs_buf.append(e)

        added_for_file = len(ids_buf) - start_idx
        print(f"[ADD-FILE] {os.path.basename(abs_path)} add={added_for_file}")

        # 6) バッチフラッシュ
        if len(ids_buf) >= BATCH_SIZE:
            col.add(ids=ids_buf, documents=docs_buf, metadatas=metas_buf, embeddings=embs_buf)
            print(f"[ADD] {len(ids_buf)} chunks (flush)")
            ids_buf, docs_buf, metas_buf, embs_buf = [], [], [], []

    # 残りをフラッシュ
    if ids_buf:
        col.add(ids=ids_buf, documents=docs_buf, metadatas=metas_buf, embeddings=embs_buf)
        print(f"[ADD] {len(ids_buf)} chunks (final)")

    print(f"[COUNT] total in collection = {col.count()}")
    print("[DONE] 登録完了")


if __name__ == "__main__":
    main()
