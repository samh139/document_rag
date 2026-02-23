"""
runtime/cli.py — v7.5 ES-Driven Runtime (GOAP vFinal)
------------------------------------------------------
Pipeline:  SLM → SlotLex → GOAP → Bandit → BT → Reward
GOAP accepts only: intent + slotlex_output
"""

import logging, json, os
from datetime import datetime
from pathlib import Path

from bt_goap_bandit.nlu.slm_nlu_adapter import SLMNLUAdapter
from bt_goap_bandit.nlp.sentiment import analyze_sentiment, _analyzer_mode
from bt_goap_bandit.slotlex import enrich_slots
from bt_goap_bandit.goap.goap_planner import build_goap_plan
from bt_goap_bandit.bt.tree import build_tree, META_INTENT_MAP
from bt_goap_bandit.bt.nodes import BB
from bt_goap_bandit.bandit import policy
from bt_goap_bandit.bandit.features import compute_reward
from model_classes.intent_agent_response import IntentAgentResponse,SLMIdentifiedSubPlan,Enriched,GoapPlan,Subplan,MetaManifest,Meta
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEMETRY_DIR = Path("telemetry")
TELEMETRY_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
def _log_event(event: dict):
    if os.getenv("BT_TELEMETRY", "").lower() not in ("1", "true", "yes"):
        return
    try:
        with open(TELEMETRY_DIR / "cli_events.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning(f"[Telemetry] Write failed: {e}")


# ---------------------------------------------------------------------------
def generate_intent_agent_response(user_input: str)->IntentAgentResponse:
    logger.info("[SLM] Decomposing user text into subplans...")
    nlu = SLMNLUAdapter()
    subplans:List[SLMIdentifiedSubPlan] = nlu.decompose(user_input)
    logger.info(f"[SLM] Found {len(subplans)} subplans")

    all_results = []

    # ==============================================================
    # LOOP THROUGH SUBPLANS
    # ==============================================================
    for i, sp in enumerate(subplans, start=1):

        text = sp.text
        intent = sp.intent
        entities = sp.entities
        conf = sp.confidence

        logger.info(f"\n🧭 Subplan {i}/{len(subplans)} — Intent={intent}, Entities={entities}")

        # ------------------------------------------------------------
        # 1️⃣ SENTIMENT
        # ------------------------------------------------------------
        sentiment = analyze_sentiment(text)
        provider = sentiment.raw.mode
        logger.info(f"[Sentiment:{sentiment}")

        # ------------------------------------------------------------
        # 2️⃣ SLOTLEX ENRICHMENT
        # ------------------------------------------------------------
        enriched:Enriched = enrich_slots(
            text= text,
            intent= intent,
            entities=entities
        )

        logger.info(f"***slotlex {enriched}")

        # ------------------------------------------------------------
        # 3️⃣ META-MANIFEST LOOKUP
        # ------------------------------------------------------------
        manifest_entry = META_INTENT_MAP.get(intent, {})
        node_type = manifest_entry.get("node_type", "generic")
        intent_family = manifest_entry.get("intent_family", node_type)
        behavioral_family = manifest_entry.get("behavioral_family", intent_family)
        default_scope = manifest_entry.get("default_scope", "contextual")
        tags = manifest_entry.get("tags", [])

        logger.info(f"[GOAP] Executing GOAP for intent='{intent}'...")

        # ------------------------------------------------------------
        # 4️⃣ BUILD GOAP PLAN → New format (tool + params)
        # ------------------------------------------------------------
        goap:GoapPlan = build_goap_plan(intent=intent, slotlex_output=enriched)

        plan = {
            "tool": goap.tool,
            "params": goap.params,
            "missing": goap.missing,
            "confidence": goap.confidence,

            # BT/Bandit compatibility layer
            "blocking_slots": goap.missing,
            "readiness_score": 1.0 if not goap.missing else 0.0,
            "missing_ratio": 0.0 if not goap.missing else 1.0,
        }

        # ------------------------------------------------------------
        # 5️⃣ CONTEXT PRIMARY — MUST BE BEFORE BANDIT SELECTION
        # ------------------------------------------------------------
        context_primary = {
            **manifest_entry,
            "intent": intent,
            "intent_family": intent_family,
            "behavioral_family": behavioral_family,
            "text": text,
            "entities": entities,
            "tags": tags,
            "sentiment_hint": sentiment.valence,
        }

        router_arm = behavioral_family  # Option A

        # ------------------------------------------------------------
        # 6️⃣ BANDIT SELECTION — Option A
        # ------------------------------------------------------------
        try:
            arm = policy.choose(
                behavioral_family=behavioral_family,
                router_arm=router_arm,
                context_primary=context_primary,
                plan=plan,
                sentiment=sentiment
            )
            logger.info(f"[Bandit] 🎯 Selected arm: {arm}")
        except Exception as e:
            logger.warning(f"[Bandit] choose() failed, fallback to router_arm: {e}")
            arm = router_arm

        # ------------------------------------------------------------
        # 7️⃣ EXECUTE BEHAVIOUR TREE
        # ------------------------------------------------------------
        bb = BB()
        bb.meta_intent = intent
        bb.entities = entities
        bb.sentiment_hint = sentiment.valence
        bb.subplans = [{"plan": plan}]

        bt_tree = build_tree({"meta_intent": intent}, bb, goap_plan=plan)
        bt_tree.tick()
        
        bt_output = getattr(bb, "signal", None)
        print("behavior tree",bt_tree)

        # ------------------------------------------------------------
        # 8️⃣ REWARD + LEARNING
        # ------------------------------------------------------------
        readiness = plan["readiness_score"]
        missing_ratio = plan["missing_ratio"]
        sent_val_num = sentiment.valence
        

        reward_val = compute_reward(
            readiness,
            missing_ratio,
            1.0 if "pos" in sent_val_num else -1.0 if "neg" in sent_val_num else 0.0
        )

        try:
            policy.learn(arm, reward_val, context_primary, plan, sentiment)
            logger.info(f"[Bandit] ✅ Learned reward={reward_val:.2f} for arm={arm}")
        except Exception as e:
            logger.warning(f"[Bandit] learn() failed: {e}")

        _log_event({
            "timestamp": datetime.utcnow().isoformat(),
            "subplan": i,
            "intent": intent,
            "arm": arm,
            "reward": reward_val,
            "blocking": plan["blocking_slots"],
        })

        # ------------------------------------------------------------
        # 9️⃣ COLLECT SUBPLAN OUTPUT
        # ------------------------------------------------------------
        all_results.append(Subplan(
            subplan_id= i,
            input= sp,
            enriched= enriched,
            goap_plan= goap,
            sentiment= sentiment,
            bandit_arm= arm,
            bt_output= bt_output,
            meta_manifest=MetaManifest(
                intent_family=intent_family,
                behavioral_family= behavioral_family,
                scope= default_scope,
                tags=tags
            )
        ))


    final=IntentAgentResponse(agent_role="signal",
                              input_text=user_input,
                              subplans=all_results,
                              meta=Meta(pipeline_version= "v7.5",
            policy_version= f"behavioral-bandit-es+{_analyzer_mode or 'hf'}",
            subplans_count= len(subplans),
            run_id= datetime.now().strftime("%Y%m%dT%H%M%SZ"),
            sentiment_provider= _analyzer_mode or "hf"))

    
    return final




   


