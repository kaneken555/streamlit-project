# 🦙 Ollama × Streamlit RAG Chatbot（日本語対応）

> ローカルLLM「Ollama」とベクトルDB「Chroma」を使った  
> 日本語対応のRAG（Retrieval-Augmented Generation）チャットアプリです。  
>
> Ollamaで実行されるLLM（例：`llama3:8b`, `qwen2:7b`, `gemma:7b`）に、  
> ローカルドキュメントから検索した知識を組み合わせて回答を生成します。

---

## 🚀 機能概要

| 機能 | 内容 |
|------|------|
| 💬 **チャットUI** | Streamlitの`st.chat_message()`を利用したChatGPT風UI |
| 🧠 **RAG検索** | Chroma + e5埋め込み（`intfloat/multilingual-e5-small`）によるドキュメント検索 |
| 📄 **ドキュメント対応** | `docs/` フォルダ内の `.txt`, `.md`, `.pdf` を自動でベクトル化 |
| 🗂️ **永続ベクトルDB** | Chromaの `PersistentClient` により `chroma_db/` に保存 |
| 🦙 **ローカルLLM連携** | Ollama経由でモデル呼び出し（`http://localhost:11434/api/chat`） |
| 🧩 **モデル選択** | サイドバーからローカルにあるOllamaモデルを選択可能 |
| 🈁 **日本語対応** | e5モデルで日本語検索、system promptで日本語回答を強制 |

---