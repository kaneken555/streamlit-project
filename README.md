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
## 🧩 ディレクトリ構成
```
streamlit-project/
├── app.py # Streamlitアプリ本体（Ollama + RAG）
├── ingest.py # ドキュメントをベクトルDBに登録するスクリプト
├── requirements.txt # 依存パッケージ
├── Makefile # （任意）便利コマンド
│
├── docs/ # 検索対象ドキュメント
│ ├── aws_saa.txt
│ └── python_basics.pdf
│
├── chroma_db/ # Chromaの永続ベクトルデータ（自動生成）
└── .streamlit/
└── secrets.toml # （任意）秘密設定（ignore推奨）
```

---
## 🛠️ セットアップ手順

### 1️⃣ 仮想環境の作成と依存インストール

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Ollama のセットアップ

macOSの場合：
```bash
brew install ollama
ollama serve
ollama pull llama3:8b
```

### 3️⃣ ドキュメントを登録（初回のみ）

docs/ にテキスト・Markdown・PDFを置きます。
```bash
python ingest.py
# → chroma_db/ にベクトルDBが生成される
```

### 4️⃣ アプリを起動
```bash
streamlit run app.py
```
ブラウザで自動的に開きます（例: http://localhost:8501）。

---
## 💡 使い方
1. 左サイドバーで Ollama の URL と モデル名 を選択
2. 「システムプロンプト」で回答方針（例: 日本語アシスタント）を指定
3. チャット欄に質問を入力
4. docs/ 内の資料を参照した回答が生成されます
5. 参考資料の出典が下部に表示されます

