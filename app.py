# app.py
# ----------------------------------------
# 🦙 Ollama × Streamlit（日本語アシスタント）
# + 最小RAG（Chroma + e5/multilingual）
# ----------------------------------------

import os
import json
import requests
import streamlit as st

# --- RAG 用 ---
import chromadb
from sentence_transformers import SentenceTransformer

# ========================================
# ページ設定
# ========================================
st.set_page_config(page_title="Ollama Chat", page_icon="🦙", layout="centered")
st.title("🦙 Ollama × Streamlit（日本語アシスタント + RAG）")

# 初期モデル（サイドバーで上書きされる）
if "model" not in st.session_state:
    st.session_state.model = "llama3:8b"

# ========================================
# RAG（Chroma + e5）初期化
# - PersistentClient でローカル永続
# - e5 は日本語に強い多言語埋め込み
# ========================================
@st.cache_resource
def get_vectordb():
    client = chromadb.PersistentClient(path="chroma_db")
    col = client.get_or_create_collection("rag_docs", metadata={"hnsw:space": "cosine"})
    embed = SentenceTransformer("intfloat/multilingual-e5-small")
    return client, col, embed

client, col, embed = get_vectordb()

def retrieve_context(query: str, top_k: int = 6):
    """e5 の推奨プレフィックスを使って検索→上位k件を連結"""
    try:
        qvec = embed.encode([f"query: {query}"], normalize_embeddings=True).tolist()[0]
        res = col.query(
            query_embeddings=[qvec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        lines, sources = [], []
        for i, (d, m) in enumerate(zip(docs, metas), 1):
            src = (m or {}).get("source", f"doc{i}")
            sources.append(str(src))
            lines.append(f"[{i}] 出典: {src}\n{d}")
        context = "\n\n---\n\n".join(lines)
        return context, sorted(set(sources))
    except Exception:
        # コレクションが空などのケースでも落とさない
        return "", []

# ========================================
# サイドバー：設定
# ========================================
with st.sidebar:
    st.header("⚙️ 設定（Ollama専用）")

    # 接続先URL（ネイティブ: http://localhost:11434）
    default_url = os.environ.get("OLLAMA_URL") or "http://localhost:11434"
    base_url = st.text_input("Ollama URL", value=default_url, help="例: http://localhost:11434")

    @st.cache_data(show_spinner=False, ttl=30)
    def list_ollama_models(url: str):
        """Ollama の /api/tags からモデル一覧を取得（失敗時は空配列）"""
        try:
            r = requests.get(url.rstrip("/") + "/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            return sorted([m["name"] for m in data.get("models", [])])
        except Exception:
            return []

    models = list_ollama_models(base_url)

    if models:
        init_idx = models.index(st.session_state.model) if st.session_state.model in models else 0
        model = st.selectbox("モデル名", models, index=init_idx)
    else:
        st.info("モデル一覧を取得できませんでした。URLや起動状態（ollama serve）を確認してください。手入力で指定できます。")
        model = st.text_input("モデル名（手入力）", value=st.session_state.model)

    # 選択/入力値を状態へ保存（下の推論で使用）
    st.session_state.model = model

    temperature = st.slider("温度 (創造性)", 0.0, 1.0, 0.2, 0.1)
    num_ctx = st.number_input(
        "コンテキスト長 (num_ctx)", min_value=2048, max_value=32768, value=8192, step=1024,
        help="長くすると過去の文脈をより保持（モデルの上限に注意）"
    )

    # 日本語アシスタントの既定 System Prompt（必要に応じて編集）
    system_prompt = st.text_area(
        "システムプロンプト",
        value=(
            "あなたは日本語で丁寧かつわかりやすく回答するアシスタントです。\n"
            "必ず日本語で答えてください。英語で出力してはいけません。\n"
            "専門用語が出る場合は日本語の補足も添えてください。\n"
            "過度に長くせず、見出しや箇条書きを適切に使って整理してください。"
        ),
        height=140,
    )

    max_history = st.number_input("履歴上限（往復数）", 2, 50, 10, 1)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 履歴クリア"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.caption("DockerのStreamlit→ネイティブOllamaは http://host.docker.internal:11434")

# ========================================
# 会話メモリ
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

def truncated_history(messages, max_rounds):
    """assistant/user の往復で上限を超えたら古いものから削る"""
    if len(messages) <= max_rounds * 2:
        return messages
    return messages[-max_rounds * 2:]

# ========================================
# Ollama 呼び出し（ストリーミング & 例外処理）
# ========================================
def call_ollama(base_url, prompt, history, model, temperature, system_prompt, num_ctx=8192):
    url = base_url.rstrip("/") + "/api/chat"

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    # 履歴（user/assistantのみ）＋今回の user
    msgs += [m for m in history if m["role"] in ("user", "assistant")]
    msgs.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "options": {
            "temperature": float(temperature),
            "num_ctx": int(num_ctx),
        },
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
    except requests.exceptions.ConnectionError:
        yield "⚠️ Ollama に接続できません。URL と起動状態(ollama serve)を確認してください。"
    except requests.exceptions.HTTPError as e:
        yield f"⚠️ HTTPエラー: {e.response.status_code} {e.response.text[:200]}"
    except Exception as e:
        yield f"⚠️ 予期せぬエラー: {type(e).__name__}: {e}"

# ========================================
# 既存履歴の描画
# ========================================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ========================================
# 入力受付 & 応答（RAG差し込みポイントあり）
# ========================================
if user_input := st.chat_input("メッセージを入力…"):
    # 日本語出力を安定させる補助行
    user_input = user_input.strip()

    # まずはユーザー発話を履歴へ
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages = truncated_history(st.session_state.messages, max_history)

    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------- ★ RAG: 前処理・検索 ★ ----------
    context, sources = retrieve_context(user_input, top_k=4)
    aug_system_prompt = (
        system_prompt
        + "\n\n# 参考資料（抜粋）\n"
        + (context or "（該当資料なし）")
        + "\n\n※上記の資料のみを根拠に、日本語で簡潔に回答してください。"
        + "\n必要に応じて [番号] を使って根拠を示してください。"
    )
    # --------------------------------------------

    with st.chat_message("assistant"):
        stream = call_ollama(
            base_url=base_url,
            prompt=user_input,                 # ユーザーの質問
            history=st.session_state.messages,
            model=st.session_state.model,      # サイドバー選択モデル
            temperature=temperature,
            system_prompt=aug_system_prompt,   # RAG文脈を同梱
            num_ctx=num_ctx,
        )
        reply = st.write_stream(stream)

    # 生成結果を会話メモリへ
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # 出典の表示（任意）
    if sources:
        st.caption("出典: " + " | ".join(sources))
