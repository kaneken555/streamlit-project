import json, requests
from typing import Iterable, List, Dict, Any
from app.core.ports.llm import LLMClient
from app.core.types import Message, ChatChunk

class OllamaClient(LLMClient):
    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            return sorted([m["name"] for m in data.get("models", [])])
        except Exception:
            return []

    def chat_stream(
        self, messages: List[Message], model: str, options: Dict[str, Any]
    ) -> Iterable[ChatChunk]:
        payload = {
            "model": model,
            "messages": [m.__dict__ for m in messages],
            "stream": True,
            "options": options or {},
        }
        url = f"{self.base_url}/api/chat"

        try:
            with requests.post(url, json=payload, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield ChatChunk(content=token)
        except requests.exceptions.ConnectionError:
            yield ChatChunk(content="⚠️ Ollama に接続できません。URL と起動状態(ollama serve)を確認してください。", done=True)
        except requests.exceptions.HTTPError as e:
            yield ChatChunk(content=f"⚠️ HTTPエラー: {e.response.status_code} {e.response.text[:200]}", done=True)
        except Exception as e:
            yield ChatChunk(content=f"⚠️ 予期せぬエラー: {type(e).__name__}: {e}", done=True)
        finally:
            yield ChatChunk(content="", done=True)
