"""
bandit/features.py — contextual feature extractor (meta-intent + behavioral-family + SlotLex aware)
v7.4 — adds GOAP→Bandit propagation, fixes missing_scope flag, enables state persistence & weighted reward.
"""

from __future__ import annotations
from typing import Dict, Tuple, List
import os, json, numpy as np
from pathlib import Path
from model_classes.intent_agent_response import Sentiment

# ---------------------------------------------------------------------------
# 🧠 Load canonical intents / behaviours
# ---------------------------------------------------------------------------
def load_meta_manifest() -> Tuple[List[str], List[str]]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "meta_intent_manifest.json"
    intents, behaviors = [], []
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        for key, val in data.items():
            if isinstance(val, dict):
                intents.append(key.lower())
                if bf := val.get("behavioral_family"):
                    behaviors.append(bf.lower())
        intents, behaviors = sorted(set(intents)), sorted(set(behaviors))
        print(f"[Bandit] ✅ Loaded {len(intents)} intents, {len(behaviors)} behavioral families from manifest.")
    except Exception as e:
        print(f"[Bandit] ⚠️ Could not load meta_intent_manifest.json: {e}")
        intents = [
            "update_record","retrieve_info","clarify_request",
            "escalate_issue","greet","smalltalk","unknown"
        ]
        behaviors = ["generic","reasoning","clarify","social"]
    return intents, behaviors

INTENTS, BEHAVIORAL_FAMILIES = load_meta_manifest()

# ---------------------------------------------------------------------------
# 🧩 SlotLex Integration (optional)
# ---------------------------------------------------------------------------
SLOTLEX_AVAILABLE = False
try:
    from bt_goap_bandit.slotlex import SlotLex
    slotlex = SlotLex()
    SLOTLEX_AVAILABLE = True
    print("[Bandit] ✅ Loaded SlotLex from bt_goap_bandit/slotlex.py")
except Exception as e:
    print(f"[Bandit] ⚠️ SlotLex unavailable: {e}")
    slotlex = None

# ---------------------------------------------------------------------------
# 🔧 Helpers
# ---------------------------------------------------------------------------
def one_hot(names: List[str], value: str | None) -> np.ndarray:
    vec = np.zeros(len(names))
    if value in names:
        vec[names.index(value)] = 1.0
    return vec

def sentiment_to_numeric(val: str | None) -> float:
    if not val: return 0.0
    val = val.lower()
    if "neg" in val: return -1.0
    if "pos" in val: return 1.0
    return 0.0

# ---------------------------------------------------------------------------
# 🎯 Feature construction
# ---------------------------------------------------------------------------
def make_features(primary: Dict, plan: Dict, sentiment: Sentiment) -> Tuple[np.ndarray, List[str]]:
    """
    Builds the contextual vector for Bandit input.
      - Intent one-hot (from GOAP/SLM)
      - Behavioural-family one-hot (optional)
      - Slot readiness / missing flags
      - Sentiment valence & intensity
    """

    intent_key = (plan.get("intent") or plan.get("meta_intent") or "").lower()
    intent_vec = one_hot(INTENTS, intent_key)

    # --- Behavioural family (from manifest) ---
    family = (plan.get("behavioral_family") or "").lower()
    behavior_vec = one_hot(BEHAVIORAL_FAMILIES, family)

    # --- Slot metrics ---
    readiness = float(plan.get("readiness_score", 0.0))
    missing_ratio = float(plan.get("missing_ratio", 1.0))
    # Fix: missing_scope flag cleared when ratio small
    missing_scope = 1.0 if missing_ratio > 0.4 else 0.0
    weighted_readiness = round(readiness * (1 - missing_ratio), 3)

    # --- Sentiment ---
    sent_val = sentiment_to_numeric(sentiment.valence)
    sent_intensity = abs(sentiment.intensity)

    vec = np.concatenate([
        intent_vec, behavior_vec,
        np.array([
            readiness, missing_ratio, missing_scope,
            weighted_readiness, sent_val, sent_intensity
        ])
    ])

    labels = (
        [f"intent_{n}" for n in INTENTS] +
        [f"behavior_{b}" for b in BEHAVIORAL_FAMILIES] +
        ["readiness","missing_ratio","missing_scope",
         "weighted_readiness","sentiment_valence","sentiment_intensity"]
    )

    return vec.astype(np.float32), labels

# ---------------------------------------------------------------------------
# 🎰 Bandit model wrapper
# ---------------------------------------------------------------------------
from bt_goap_bandit.bandit.model import LinUCBBandit
import json

class BanditModel:
    """Wrapper for LinUCB using contextual features and persistent state."""
    STATE_FILE = ".bandit_state.json"

    def __init__(self):
        self.arms = ["open","empathetic_open","clarify_scope","probe_missing","direct_execute"]
        d = len(INTENTS) + len(BEHAVIORAL_FAMILIES) + 6
        self.bandit = LinUCBBandit(arms=self.arms, d=d, alpha=0.8)
        self.intent_index: Dict[str,int] = {}
        self.load_state()

    # ------------------------
    def _encode_context(self, ctx: Dict) -> np.ndarray:
        intent = ctx.get("intent","unknown")
        if intent not in self.intent_index:
            self.intent_index[intent] = len(self.intent_index)+1
        intent_id = self.intent_index[intent]
        conf = float(ctx.get("confidence",0.5))
        valence = ctx.get("sent_valence","neutral")
        intensity = float(ctx.get("sent_intensity",0.0))
        missing = len(ctx.get("missing_slots",[]))
        val_map = {"positive":1.0,"neutral":0.0,"negative":-1.0}
        val_score = val_map.get(valence,0.0)
        return np.array([intent_id,conf,val_score,intensity,missing],dtype=float)

    def select_arm(self, ctx: Dict) -> str:
        x = self._encode_context(ctx)
        return self.bandit.predict(x)

    def update(self, chosen_arm: str, reward: float, ctx: Dict) -> None:
        x = self._encode_context(ctx)
        self.bandit.update(chosen_arm, reward, x)
        self.save_state()

    # ------------------------
    # 💾 Persistence
    # ------------------------
    def save_state(self):
        try:
            state = self.bandit.export_state()
            with open(self.STATE_FILE,"w") as f:
                json.dump(state,f)
            print(f"💾 [Bandit] State saved → {self.STATE_FILE}")
        except Exception as e:
            print(f"⚠️ [Bandit] Save failed: {e}")

    def load_state(self):
        if not os.path.exists(self.STATE_FILE): return
        try:
            with open(self.STATE_FILE,"r") as f:
                state = json.load(f)
            self.bandit.import_state(state)
            print(f"💾 [Bandit] State loaded ← {self.STATE_FILE}")
        except Exception as e:
            print(f"⚠️ [Bandit] Load failed: {e}")

# ---------------------------------------------------------------------------
# 🧮 Reward helper
# ---------------------------------------------------------------------------
def compute_reward(readiness: float, missing_ratio: float, sentiment_valence: float) -> float:
    """Composite reward balancing readiness, slot completion, and empathy."""
    alpha, beta, gamma = 0.6, 0.25, 0.15
    return (
        alpha * readiness +
        beta * (1 - missing_ratio) +
        gamma * max(0.0, -sentiment_valence)
    )
