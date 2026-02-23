from __future__ import annotations
import argparse, json, os, sys
from typing import Any, Dict, Optional

# Defaults (overridable via env)
os.environ.setdefault("BT_BANDIT", "linucb")
os.environ.setdefault("BT_BANDIT_PATH", ".bandit_state.json")
os.environ.setdefault("BT_BANDIT_DECAY", "0.0")

from bt_goap_bandit.bandit.policy import get_policy  # noqa: E402

# Optional telemetry
try:
    from bt_goap_bandit.telemetry.metrics import reward_from_outcome  # noqa: E402
    from bt_goap_bandit.telemetry.log import JsonlLogger  # noqa: E402
    HAVE_TELEMETRY = True
except Exception:
    HAVE_TELEMETRY = False

# Optional counters
try:
    from bt_goap_bandit.telemetry.counters import update_counters  # noqa: E402
    HAVE_COUNTERS = True
except Exception:
    HAVE_COUNTERS = False


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_mapping(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    # Try JSON first
    try:
        return json.loads(txt)
    except Exception:
        # Simple YAML-ish "k: v" pairs
        mapping: Dict[str, float] = {}
        for line in txt.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    mapping[k.strip()] = float(v.strip())
                except Exception:
                    pass
        return mapping


def _reward_from_label(label: Optional[str], mapping: Dict[str, float]) -> Optional[float]:
    if not label:
        return None
    label = str(label).strip().lower()
    mapping = mapping or {
        "clicked": 1.0,
        "continued": 0.8,
        "thanks": 0.6,
        "ok": 0.6,
        "abandoned": 0.0,
        "negative": 0.0,
    }
    return mapping.get(label)


def main(argv=None):
    p = argparse.ArgumentParser(description="Bandit learning endpoint (v2.1)")
    # New: context bundle
    p.add_argument("--context", help="JSON with keys: primary, plan, sentiment, bandit.arm")
    # Individual pieces (override context if provided)
    p.add_argument("--primary", help="path or '-' for stdin")
    p.add_argument("--plan", help="path to plan json")
    p.add_argument("--sentiment", help="path to sentiment json")
    # Outcome event (for auto reward and/or arm)
    p.add_argument("--event", help="OutcomeEvent json (path or '-')")
    # Arm controls
    p.add_argument("--arm", help="explicit arm")
    p.add_argument("--auto-arm", action="store_true", help="infer arm from --event or --context")
    # Reward controls
    p.add_argument("--reward", type=float, help="direct reward in [0,1]")
    p.add_argument("--reward-label", help="semantic label (clicked/continued/thanks/abandoned/negative)")
    p.add_argument("--reward-map", help="JSON/YAML mapping for --reward-label → score")
    # Safety / output
    p.add_argument("--force", action="store_true", help="learn even if explicit --arm mismatches event/context arm")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    # Load context bundle first
    ctx = _load_json(args.context) if args.context else {}
    ctx_primary = ctx.get("primary") or {}
    ctx_plan = ctx.get("plan") or {}
    ctx_sent = ctx.get("sentiment") or {}
    ctx_bandit_arm = None
    try:
        ctx_bandit_arm = (ctx.get("bandit") or {}).get("arm")
    except Exception:
        ctx_bandit_arm = None

    # Then load explicit pieces (explicit overrides context)
    primary = _load_json(args.primary) if args.primary else ctx_primary
    plan = _load_json(args.plan) if args.plan else ctx_plan
    sentiment = _load_json(args.sentiment) if args.sentiment else ctx_sent

    # Load event (for auto-reward and/or arm)
    event = _load_json(args.event) if args.event else None
    event_arm = None
    if event:
        event_arm = event.get("arm") or (event.get("context") or {}).get("arm")

    # --- Arm resolution priority ---
    # 1) explicit --arm (with mismatch guard vs event/context if present, unless --force)
    # 2) if --auto-arm: prefer event_arm, else ctx_bandit_arm
    # 3) else: use event_arm or ctx_bandit_arm if present
    # 4) fallback heuristic from plan
    arm = args.arm
    if arm:
        # guard only when user explicitly provided --arm
        ref_arm = event_arm or ctx_bandit_arm
        if ref_arm and arm != ref_arm and not args.force:
            raise SystemExit(
                f"[learn] arm mismatch: provided --arm={arm!r} vs observed={ref_arm!r}. Use --force to override."
            )
    else:
        if args.auto_arm:
            arm = event_arm or ctx_bandit_arm
        else:
            arm = event_arm or ctx_bandit_arm
        if not arm:
            arm = "chips" if "scope" in (plan.get("blocking_slots") or []) else "empathetic_open"

    # --- Reward computation ---
    r: Optional[float] = None
    if args.reward is not None:
        r = float(args.reward)
    elif args.reward_label is not None:
        r = _reward_from_label(args.reward_label, _load_mapping(args.reward_map))
    elif event is not None and HAVE_TELEMETRY:
        r = reward_from_outcome(event)
    if r is None:
        r = 0.0
    r = max(0.0, min(1.0, float(r)))

    # Learn
    policy = get_policy()
    decay = float(os.getenv("BT_BANDIT_DECAY", "0.0") or "0.0")
    try:
        policy.learn(arm, r, primary, plan, sentiment, decay=decay)
    except TypeError:
        policy.learn(arm, r, primary, plan, sentiment)

    # Telemetry log (best effort)
    if HAVE_TELEMETRY:
        try:
            JsonlLogger().write({
                "type": "learn",
                "arm": arm,
                "reward": r,
                "primary": primary,
                "plan": plan,
                "sentiment": sentiment,
                "event": event,
                "context_used": bool(args.context),
            })
        except Exception:
            pass

    # Counters (best effort)
    if HAVE_COUNTERS:
        try:
            readiness = "ready" if plan.get("ready_to_retrieve") else "blocked" if plan.get("blocking_slots") else "unknown"
            s = sentiment or {}
            sent_bucket = "pos" if (s.get("compound", 0) > 0.05) else "neg" if (s.get("compound", 0) < -0.05) else "neu"
            update_counters(arm=arm, reward=r, readiness_bucket=readiness, sentiment_bucket=sent_bucket)
        except Exception:
            pass

    if not args.quiet:
        print(json.dumps({
            "ok": True,
            "arm": arm,
            "reward": r,
            "state_path": os.getenv("BT_BANDIT_PATH", ".bandit_state.json"),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
