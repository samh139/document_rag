from __future__ import annotations
import json, os
from typing import Dict, Any

COUNTERS_PATH = os.getenv("BT_BANDIT_COUNTERS", ".bandit_counters.json")

def _read(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "arms": {},
            "reward_hist": {"0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.8":0,"0.8-1.0":0},
            "readiness": {"ready":0,"blocked":0,"unknown":0},
            "sentiment": {"neg":0,"neu":0,"pos":0}
        }

def _write(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _bucketize(r: float) -> str:
    if r < 0.2: return "0-0.2"
    if r < 0.4: return "0.2-0.4"
    if r < 0.6: return "0.4-0.6"
    if r < 0.8: return "0.6-0.8"
    return "0.8-1.0"

def update_counters(*, arm: str, reward: float, readiness_bucket: str, sentiment_bucket: str) -> None:
    c = _read(COUNTERS_PATH)
    c["arms"].setdefault(arm, {"count":0, "sum_reward":0.0})
    c["arms"][arm]["count"] += 1
    c["arms"][arm]["sum_reward"] += float(reward)
    c["reward_hist"][_bucketize(reward)] += 1
    if readiness_bucket in c["readiness"]:
        c["readiness"][readiness_bucket] += 1
    if sentiment_bucket in c["sentiment"]:
        c["sentiment"][sentiment_bucket] += 1
    _write(COUNTERS_PATH, c)
