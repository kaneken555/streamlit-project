import streamlit as st
from typing import List
from app.core.types import Message
from app.services.chat_orchestrator import ChatOrchestrator
from app.registry.providers import build_stack
from app.config.settings import settings

st.set_page_config(page_title="Chat + RAG", page_icon="🦙", layout="centered")
st.title("🦙 Ollama × Streamlit（日本語アシスタント + RAG）")

# 会話メモリ（UI層では dict で保持 → 呼ぶ直前に Message 化でもOK）
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = settings.default_model

def to_messages_dicts_to_Message(lst: List[dict]) -> List[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in lst]

def truncated_history(messages: List[dict], max_rounds: int):
    if len(messages) <= max_rounds * 2:
        return messages
    return messages[-max_rounds * 2:]

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定（Ollama）")
    base_url = st.text_input("Ollama URL", value=settings.ollama_url, help="例: http://localhost:11434")

    # ProviderとRetrieverを構築（キャッシュ）
    @st.cache_resource(show_spinner=False)
    def get_stack(url: str):
        llm, retriever = build_stack("ollama", base_url=url, embed_model=settings.embed_model, chroma_path=settings.chroma_path)
        return llm, retriever

    llm, retriever = get_stack(base_url)

    try:
        models = llm.list_models()
    except Exception:
        models = []

    if models:
        init_idx = models.index(st.session_state.model) if st.session_state.model in models else 0
        model = st.selectbox("モデル名", models, index=init_idx)
    else:
        st.info("モデル一覧を取得できませんでした。手入力で指定できます。")
        model = st.text_input("モデル名（手入力）", value=st.session_state.model)

    st.session_state.model = model

    temperature = st.slider("温度 (創造性)", 0.0, 1.0, settings.temperature, 0.1)
    num_ctx = st.number_input("コンテキスト長 (num_ctx)", 2048, 32768, settings.num_ctx, 1024)

    system_prompt = st.text_area(
        "システムプロンプト",
        value=("あなたは日本語で丁寧かつわかりやすく回答するアシスタントです。\n"
               "必ず日本語で答えてください。英語で出力してはいけません。\n"
               "専門用語が出る場合は日本語の補足も添えてください。\n"
               "過度に長くせず、見出しや箇条書きを適切に使って整理してください。"),
        height=140,
    )

    max_history = st.number_input("履歴上限（往復数）", 2, 50, 10, 1)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 履歴クリア"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.caption("Docker→ネイティブOllama: http://host.docker.internal:11434")

# 既存履歴の描画
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# オーケストレータ（毎回再生成でも軽い）
orch = ChatOrchestrator(llm=llm, retriever=retriever, base_system_prompt=system_prompt)

# 入力受付
if user_input := st.chat_input("メッセージを入力…"):
    user_input = user_input.strip()

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages = truncated_history(st.session_state.messages, max_history)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        stream, sources = orch.run_stream(
            user_input=user_input,
            history=to_messages_dicts_to_Message([m for m in st.session_state.messages if m["role"] in ("user","assistant")]),
            model=st.session_state.model,
            options={"temperature": float(temperature), "num_ctx": int(num_ctx)},
            top_k=4,
        )
        # Streamlit の write_stream はテキストイテレータを受け取る
        reply = st.write_stream((chunk.content for chunk in stream))

    st.session_state.messages.append({"role": "assistant", "content": reply})

    if sources:
        st.caption("出典: " + " | ".join(sources))
