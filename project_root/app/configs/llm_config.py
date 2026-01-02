# app/configs/llm_config.py

import os
import json
import requests
import re
from typing import Dict

# Ollama base
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Models
ENGAGEMENT_MODEL = os.getenv("ENGAGEMENT_MODEL", "gemma3:4b")
QUERY_REFINER_MODEL = os.getenv("QUERY_REFINER_MODEL", "gemma3:4b")
RAG_SYNTHESIS_MODEL = os.getenv("RAG_SYNTHESIS_MODEL", "gemma3:8b")
STM_SUMMARY_MODEL = os.getenv("STM_SUMMARY_MODEL", "gemma3:4b")

# Defaults
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

def fire_fast_modal_request_chat_for_force_json(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": STM_SUMMARY_MODEL,          # ✅ correct key
        "prompt": f"{system_prompt}\n\n{user_prompt}",  # ✅ merged prompt
        "stream": False,
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def extract_json_from_llm_output(text: str):
    """
    Convert noisy model output into a Python dict or list.

    - Removes code fences and enclosing quotes.
    - Extracts the first JSON object/array in the text.
    - Escapes unescaped control characters that appear inside JSON string literals
      (literal newlines, tabs, other control chars) so json.loads will accept them.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    s = text.strip()

    # 1) Remove fenced code block markers (```json ... ```)
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s, flags=re.IGNORECASE)

    # 2) Remove surrounding quotes if whole thing is a quoted JSON string
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        inner = s[1:-1]
        # heuristic: only strip if it looks like escaped JSON
        if inner.count('\\') >= 2:
            s = inner

    # 3) Try a unicode-unescape pass if heavily escaped (e.g. the whole JSON was double-escaped)
    if '\\n' in s or '\\"' in s or "\\t" in s:
        try:
            s2 = bytes(s, "utf-8").decode("unicode_escape")
            # prefer s2 if it looks like JSON
            if ("{" in s2 and "}" in s2) or ("[" in s2 and "]" in s2):
                s = s2
        except Exception:
            # ignore failures in unescaping
            pass

    # 4) Extract substring from first { or [ to last } or ]
    start_obj, start_arr = s.find('{'), s.find('[')
    if start_obj == -1 and start_arr == -1:
        raise ValueError("No JSON object or array found in string")

    if start_obj == -1:
        start = start_arr
    elif start_arr == -1:
        start = start_obj
    else:
        start = min([i for i in (start_obj, start_arr) if i != -1])

    end_obj, end_arr = s.rfind('}'), s.rfind(']')
    end = max(end_obj, end_arr)
    if end == -1 or end < start:
        raise ValueError("Couldn't find matching JSON end '}' or ']'")

    s = s[start:end+1]

    # 5) Escape unescaped control chars that appear INSIDE JSON string literals
    def escape_control_chars_in_json_string(js: str) -> str:
        out = []
        in_str = False
        escape = False
        for ch in js:
            if not in_str:
                # not inside a string literal
                out.append(ch)
                if ch == '"':
                    in_str = True
                    escape = False
                continue

            # we're inside a JSON string literal here
            if escape:
                # previous char was backslash - keep escaped char as-is
                out.append(ch)
                escape = False
                continue

            if ch == '\\':
                out.append(ch)
                escape = True
                continue

            if ch == '"':
                out.append(ch)
                in_str = False
                continue

            # if ch is a control character, escape it
            code = ord(ch)
            if ch == '\n':
                out.append('\\n')
            elif ch == '\r':
                out.append('\\r')
            elif ch == '\t':
                out.append('\\t')
            elif code < 0x20:
                # other control characters -> use \uXXXX
                out.append('\\u%04x' % code)
            else:
                out.append(ch)

        return ''.join(out)

    cleaned = escape_control_chars_in_json_string(s)

    # 6) Final parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # include a short preview of what we tried to parse to help debugging
        preview = cleaned[:1000] + ("..." if len(cleaned) > 1000 else "")
        raise json.JSONDecodeError(
            msg=f"{e.msg}. After cleaning, preview: {preview}",
            doc=cleaned,
            pos=e.pos
        ) from e


def fire_fast_modal_request_chat_get_dict(system_prompt: str, user_prompt: str) -> Dict:
    text = fire_fast_modal_request_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    return extract_json_from_llm_output(text)


def fire_fast_modal_request_chat(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": STM_SUMMARY_MODEL,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False,
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    # ✅ ALWAYS return only clean text
    return data.get("response", "").strip()
