"""
bandit/policy.py — Option A (Zero-Heuristic, Router-First)
---------------------------------------------------------------
✓ Router → behavioral_family → bandit_arm = FINAL ARM
✓ Bandit NEVER overrides router
✓ FULL ARM LIST synced with all behavioral_families in your system
✓ Safe index-based learning only
✓ Deterministic, stable, no sentiment/readiness heuristics
---------------------------------------------------------------
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

from bt_goap_bandit.bandit.model import LinUCBBandit
from bt_goap_bandit.bandit import features
from model_classes.intent_agent_response import Sentiment

# =====================================================================
# 🔥 **FULL + PATCHED ARM SET**
# MUST MATCH ALL behavioral_family VALUES IN:
#    • meta_intent_manifest.json
#    • router output
#    • test suite results
# =====================================================================
ARMS = [
    "open",
    "clarify_scope",
    "probe_missing",
    "direct_execute",
    "data_probe",
    "analytical_reasoning",
    "clarification_request",
    "neutral_smalltalk",
    "task_execution",
    "exploratory_reasoning",
    "reflective_reasoning",
    "frustration_expression",
    "system_control",
    "conversation_withdrawal",
    "hesitant_disagreement"
]

# Persisted learning state
STATE_FILE = Path(".bandit_state.json")
TELEMETRY_DIR = Path("telemetry")
TELEMETRY_DIR.mkdir(exist_ok=True, parents=True)


# =====================================================================
# 📡 OPTIONAL TELEMETRY
# =====================================================================
def _telemetry_log(rec: Dict):
    if os.getenv("BT_TELEMETRY", "").lower() not in ("1", "true", "yes"):
        return

    try:
        with open(TELEMETRY_DIR / "bandit_events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except:
        pass


# =====================================================================
# 🎯 SINGLETON POLICY INSTANCE
# =====================================================================
_POLICY: LinUCBBandit | None = None


def get_policy(dim: int = 16) -> LinUCBBandit:
    """
    Loads LinUCB once and restores state if available.
    """
    global _POLICY

    if _POLICY is None:
        _POLICY = LinUCBBandit(n_arms=len(ARMS), dim=dim, alpha=0.3)

        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    _POLICY.import_state(json.load(f))
                print(f"[Bandit] ♻️ State restored ← {STATE_FILE}")
            except Exception as e:
                print(f"[Bandit] ⚠️ Failed to load state: {e}")

        print(f"[Bandit] ✅ LinUCB ready | dim={dim} | arms={ARMS}")

    return _POLICY


# =====================================================================
# 🎛️ ARM SELECTION: ROUTER-FIRST (NO HEURISTICS)
# =====================================================================
def choose(behavioral_family: str,
           router_arm: str,
           context_primary: Dict,
           plan: Dict,
           sentiment: Sentiment) -> str:
    """
    Deterministic:
        router_arm → FINAL arm
    LinUCB advisory NEVER overrides.
    """

    final_arm = router_arm

    # Optional advisory mode only (never overrides)
    use_bandit = os.getenv("BT_BANDIT", "").lower() in ("1", "true", "yes", "linucb")

    if use_bandit:
        try:
            x, _ = features.make_features(context_primary, plan, sentiment)
            policy = get_policy(dim=len(x))
            _ = policy.predict(x)   # advisory only
        except Exception as e:
            print(f"[Bandit] ⚠️ LinUCB advisory failed: {e}")

    # Telemetry
    _telemetry_log({
        "event": "choose",
        "behavioral_family": behavioral_family,
        "router_arm": router_arm,
        "final_arm": final_arm,
        "timestamp": datetime.utcnow().isoformat()
    })

    return final_arm


# =====================================================================
# 📈 ONLINE LEARNING — INDEX-BASED & SAFE
# =====================================================================
def learn(arm: str, reward: float,
          context_primary: Dict,
          plan: Dict,
          sentiment: Sentiment) -> None:
    """
    Safe learning:
        • Only updates if arm exists
        • Always index-based
    """

    if arm not in ARMS:
        print(f"[Bandit] ⚠️ Unknown arm '{arm}', skipping update.")
        return

    try:
        arm_idx = ARMS.index(arm)
        x, _ = features.make_features(context_primary, plan, sentiment)

        policy = get_policy(dim=len(x))
        policy.update(arm_idx, reward, x)

        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(policy.export_state(), f, indent=2)

        print(f"[Bandit] ✅ Learn({arm}) → r={reward:.2f}")

    except Exception as e:
        print(f"[Bandit] ⚠️ learn() failed: {e}")
