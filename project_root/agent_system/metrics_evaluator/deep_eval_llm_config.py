from deepeval.models.base_model import DeepEvalBaseLLM
import asyncio
from deepeval.metrics.faithfulness.schema import Truths
import json
from typing import Any, Dict, Optional
import requests


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "llama3.1:8b"   # change to your local Ollama model


def fire_ollama_request(
    prompt: str,
    model_name: str,
    schema: Optional[type] = None,
    timeout: int = 120,
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
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._kwargs = kwargs

    def load_model(self):
        """Load/initialize the underlying client or model (sync)."""
        self._client = "INITED"   # placeholder

    def get_model_name(self) -> str:
        """Return a human/model id string."""
        return self._model_name

    async def a_generate(self, prompt: str, **kwargs) -> str | dict:
    # expand kwargs properly
        return self._generate_reuse(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        print("kwargs in generate:", kwargs)
        # expand kwargs properly
        return self._generate_reuse(prompt, **kwargs)

    def _generate_reuse(self, prompt: str, **kwargs):
        result = fire_ollama_request(prompt=prompt)
        print("kwargs before fetch schema:", kwargs)
        schema = kwargs.get("schema")
        if schema:
            # assuming schema is a Pydantic model (like Truths, Claims, etc.)
            obj = schema.model_validate(result)
            print("created object=", obj)
            return obj

        return result