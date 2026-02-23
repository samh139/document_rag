"""
bt_goap_bandit/bandit_eval.py
---------------------------------------------------------------
Batch-tests Bandit policy predictions using a JSON test set.

Input  : /Users/yogeshg/Documents/PY/bt_goap_bandit_v4/bandit_testset_neutral.json
Output : telemetry/bandit_eval_results.jsonl  (append mode)
---------------------------------------------------------------
"""

import json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

# --- Paths ---
BASE = Path(__file__).resolve().parent.parent
TEST_FILE = BASE / "bandit_testset_neutral.json"
LOG_FILE = BASE / "telemetry" / "bandit_eval_results.jsonl"
LOG_FILE.parent.mkdir(exist_ok=True)

# --- Load test cases ---
with open(TEST_FILE, "r", encoding="utf-8") as f:
    TEST_CASES = json.load(f)

print(f"[bandit_eval] Loaded {len(TEST_CASES)} test cases from {TEST_FILE}")


def run_query(text: str):
    """Run CLI for a single text and return (arm, sentiment, error)."""
    cmd = ["python", "-m", "bt_goap_bandit.runtime.cli", "--json", text]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout.strip()

    # 🧩 Try to extract the last JSON block from mixed logs
    json_block = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_block = line
            break

    if not json_block:
        print("[debug] No JSON detected, sample tail:\n", "\n".join(stdout.splitlines()[-5:]))
        return "error", "n/a", "no_json_output"

    try:
        data = json.loads(json_block)
        sub = data.get("subplans", [{}])[0]
        arm = sub.get("bandit_arm", "unknown")
        sentiment = sub.get("sentiment", {}).get("valence", "unknown")
        return arm, sentiment, None
    except Exception as e:
        return "error", "n/a", f"parse_error: {e}"


# --- Run evaluation ---
results = []
start = time.time()

for i, case in enumerate(TEST_CASES, 1):
    arm, sentiment, err = run_query(case["text"])
    match = (arm == case["expected_arm"])
    status = "✅" if match else "❌"
    print(f"[{i:02}] {status} {case['text']} → {arm} (exp={case['expected_arm']})")

    results.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "idx": i,
        "text": case["text"],
        "pred_arm": arm,
        "expected_arm": case["expected_arm"],
        "match": match,
        "sentiment": sentiment,
        "error": err,
    })

# --- Save results ---
with open(LOG_FILE, "a", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

elapsed = time.time() - start
total = len(results)
matches = sum(r["match"] for r in results)
print(f"\nSummary: {matches}/{total} correct ({matches/total*100:.1f}%) in {elapsed:.1f}s")
print(f"Results saved → {LOG_FILE}")
