from typing import Any, Dict, Optional
import json
import requests
from deepeval.models.base_model import DeepEvalBaseLLM

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "gemma3:12b"


def fire_ollama_request(
    prompt: str,
    model_name: str,
    timeout: int = 1000,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    raw_text = data.get("response", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"response": raw_text}


class MyLLMWrapper(DeepEvalBaseLLM):
    def __init__(self, model_name: str = OLLAMA_MODEL_NAME, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._kwargs = kwargs

    def load_model(self):
        self._client = "OLLAMA"

    def get_model_name(self) -> str:
        return self._model_name

    async def a_generate(self, prompt: str, **kwargs):
        return self._generate_reuse(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs):
        return self._generate_reuse(prompt, **kwargs)

    def _generate_reuse(self, prompt: str, **kwargs):
        result = fire_ollama_request(
            prompt=prompt,
            model_name=self._model_name,
        )

        schema = kwargs.get("schema")
        if schema:
            return schema.model_validate(result)

        return result