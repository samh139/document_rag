from __future__ import annotations
import argparse, json, sys

def iter_jsonl(path: str):
    f = sys.stdin if path == "-" else open(path, "r", encoding="utf-8")
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue
    if f is not sys.stdin:
        f.close()

def main(argv=None):
    p = argparse.ArgumentParser(description="Replay learn events from NDJSON")
    p.add_argument("jsonl", help="path to .jsonl (or '-')")
    args = p.parse_args(argv)

    from bt_goap_bandit.runtime.learn import main as learn_main
    n = 0
    for rec in iter_jsonl(args.jsonl):
        if rec.get("type") == "learn":
            evt = rec.get("event") or {"arm": rec.get("arm"), "score": rec.get("reward")}
        else:
            evt = rec
        sys.stdin = sys.__stdin__ = open(os.devnull)
        learn_main(["--event", "-", "--quiet"])
        n += 1
    print(json.dumps({"ok": True, "replayed": n}))

if __name__ == "__main__":
    main()
