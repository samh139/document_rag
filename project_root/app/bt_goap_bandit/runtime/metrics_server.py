from __future__ import annotations
import os, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

COUNTERS_PATH = os.getenv("BT_BANDIT_COUNTERS", ".bandit_counters.json")
STATE_PATH = os.getenv("BT_BANDIT_PATH", ".bandit_state.json")

def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _exposition_text(counters: Dict[str, Any], state: Dict[str, Any]) -> str:
    lines = []
    lines.append("# HELP bt_bandit_up 1 if server is running")
    lines.append("# TYPE bt_bandit_up gauge")
    lines.append("bt_bandit_up 1")

    arms = counters.get("arms", {})
    lines.append("# HELP bt_bandit_arm_count Number of learns per arm")
    lines.append("# TYPE bt_bandit_arm_count counter")
    lines.append("# HELP bt_bandit_arm_avg_reward Average reward per arm")
    lines.append("# TYPE bt_bandit_arm_avg_reward gauge")
    for arm, stats in arms.items():
        count = float(stats.get("count", 0))
        sum_reward = float(stats.get("sum_reward", 0.0))
        avg = (sum_reward / count) if count > 0 else 0.0
        lines.append(f'bt_bandit_arm_count{{arm="{arm}"}} {count}')
        lines.append(f'bt_bandit_arm_avg_reward{{arm="{arm}"}} {avg}')

    rh = counters.get("reward_hist", {})
    lines.append("# HELP bt_bandit_reward_bucket Count of rewards by bucket")
    lines.append("# TYPE bt_bandit_reward_bucket counter")
    for bucket, val in rh.items():
        lines.append(f'bt_bandit_reward_bucket{{bucket="{bucket}"}} {float(val)}')

    rd = counters.get("readiness", {})
    lines.append("# HELP bt_bandit_readiness_count Count of events by readiness bucket")
    lines.append("# TYPE bt_bandit_readiness_count counter")
    for k, v in rd.items():
        lines.append(f'bt_bandit_readiness_count{{bucket="{k}"}} {float(v)}')

    sb = counters.get("sentiment", {})
    lines.append("# HELP bt_bandit_sentiment_count Count of events by sentiment bucket")
    lines.append("# TYPE bt_bandit_sentiment_count counter")
    for k, v in sb.items():
        lines.append(f'bt_bandit_sentiment_count{{bucket="{k}"}} {float(v)}')

    if state:
        try:
            arms_in_state = len(state.get("arms", {}))
            lines.append("# HELP bt_bandit_arms Tracked arms in state")
            lines.append("# TYPE bt_bandit_arms gauge")
            lines.append(f"bt_bandit_arms {float(arms_in_state)}")
        except Exception:
            pass

    return "\n".join(lines) + "\n"

class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, body: Dict[str, Any]):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/healthz":
            self._json(200, {"ok": True})
            return

        if self.path == "/metrics":
            counters = _read_json(COUNTERS_PATH)
            state = _read_json(STATE_PATH)
            body = _exposition_text(counters, state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/state":
            counters = _read_json(COUNTERS_PATH)
            state = _read_json(STATE_PATH)
            # small, opinionated digest
            arms_digest = {}
            for arm, s in (counters.get("arms") or {}).items():
                c = float(s.get("count", 0))
                sr = float(s.get("sum_reward", 0.0))
                arms_digest[arm] = {"count": c, "avg_reward": (sr / c) if c > 0 else 0.0}
            self._json(200, {
                "counters": {
                    "arms": arms_digest,
                    "reward_hist": counters.get("reward_hist", {}),
                    "readiness": counters.get("readiness", {}),
                    "sentiment": counters.get("sentiment", {}),
                },
                "state": state,  # raw bandit state for inspection
            })
            return

        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/reset-counters":
            # reset counters only (safe for dev)
            empty = {
                "arms": {},
                "reward_hist": {"0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.8":0,"0.8-1.0":0},
                "readiness": {"ready":0,"blocked":0,"unknown":0},
                "sentiment": {"neg":0,"neu":0,"pos":0},
            }
            try:
                _write_json(COUNTERS_PATH, empty)
                self._json(200, {"ok": True, "reset": "counters"})
            except Exception as e:
                self._json(500, {"ok": False, "error": str(e)})
            return

        self._json(404, {"error": "not found"})

def main():
    host = os.getenv("METRICS_HOST", "0.0.0.0")
    port = int(os.getenv("METRICS_PORT", "9108"))
    httpd = HTTPServer((host, port), Handler)
    print(f"[metrics] serving on http://{host}:{port}  (Counters: {COUNTERS_PATH}, State: {STATE_PATH})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
