from typing import Any, Dict
import json
import re
import requests

from deepeval.models.base_model import DeepEvalBaseLLM

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_NAME = "gemma3:12b"


def fire_ollama_request(
    prompt: str,
    model_name: str,
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
    return response.json()


def _strip_code_fences(text: str) -> str:
    text = text.strip()

    # remove ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


class MyLLMWrapper(DeepEvalBaseLLM):
    def __init__(self, model_name: str = OLLAMA_MODEL_NAME, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._kwargs = kwargs

    def load_model(self):
        self._client = "OLLAMA"

    def get_model_name(self) -> str:
        return self._model_name

    async def a_generate(self, prompt: str, **kwargs: Any):
        return self._generate_reuse(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs: Any):
        return self._generate_reuse(prompt, **kwargs)

    def _generate_reuse(self, prompt: str, **kwargs: Any):
        result = fire_ollama_request(
            prompt=prompt,
            model_name=self._model_name,
        )

        schema = kwargs.get("schema")

        # Ollama usually returns text in result["response"]
        raw_text = result.get("response", "")
        raw_text = _strip_code_fences(raw_text)

        if schema:
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse model output as JSON for schema {schema.__name__}. "
                    f"Raw output was: {raw_text}"
                ) from e

            return schema.model_validate(parsed)

        return raw_text