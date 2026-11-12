#!/usr/bin/env python3
"""Zero-shot accuracy and calibration of a running cev server on small tasks,
per debias mode (stdlib only).

    python3 scripts/eval_zero_shot.py http://127.0.0.1:8080 [REFERENCE_URL]

Items live in eval/items.jsonl: ticket severity (asked as a 3-level score and
as "is the user blocked?") and ticket routing. For every mode it prints
accuracy per question, then over all answers pooled: log loss, Brier score and
expected calibration error (ECE, equal-width bins on the top probability).
Lower is better for those three; a calibrated model has ECE near 0.

With a second server (e.g. the f32 model next to a quantized one) it also
reports how closely the first tracks it: how often they pick the same answer
and the mean and largest difference in any probability.
"""

import argparse
import json
import math
import os
import urllib.request

LEVELS = ["Cosmetic; no impact", "Degraded, workaround exists", "Blocking; no workaround"]
TEAMS = {"billing": "Payments, invoices, refunds", "tech": "Bugs, crashes, errors", "sales": "Pricing, new purchases, upgrades"}
MODES = ["none", "calibrate", "permute", "full"]
HERE = os.path.dirname(os.path.abspath(__file__))


def post(base, body):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def probs(a):
    return [1 - a["noul"], a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())


def log_loss(p, target):
    return -math.log(max(p[target], 1e-12))


def brier(p, target):
    return sum((v - (i == target)) ** 2 for i, v in enumerate(p))


def reliability(samples, bins):
    """Equal-width bins over (top probability, top option was right)."""
    out = [[0, 0.0, 0.0] for _ in range(bins)]
    for conf, right in samples:
        b = out[min(int(conf * bins), bins - 1)]
        b[0] += 1
        b[1] += conf
        b[2] += right
    return [(n, c / n, r / n) for n, c, r in out if n]


def ece(samples, bins=15):
    n = len(samples)
    return sum(k / n * abs(c - r) for k, c, r in reliability(samples, bins)) if n else 0.0


def load_items(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8080")
    ap.add_argument("ref", nargs="?", help="second server to compare answers against")
    ap.add_argument("--items", default=os.path.join(HERE, "..", "eval", "items.jsonl"))
    ap.add_argument("--bins", type=int, default=15, help="ECE bins")
    ap.add_argument("--modes", default=",".join(MODES), help="comma-separated debias modes")
    args = ap.parse_args()
    items = load_items(args.items)
    sev = [i for i in items if i["kind"] == "severity"]
    route = [i for i in items if i["kind"] == "route"]

    print(f"{len(sev)} severity + {len(route)} routing items; ECE over {args.bins} bins, pooled over all answers")
    for mode in args.modes.split(","):
        hits = {"severity": 0, "blocked": 0, "route": 0}
        ms, same, delta, preds = [], [], [], []

        def ask(state, questions):
            body = {"state": state, "questions": questions, "no_store": True, "debias": mode}
            r = post(args.base, body)
            if args.ref:
                for qid, a in post(args.ref, body)["answers"].items():
                    p, q = probs(r["answers"][qid]), probs(a)
                    same.append(p.index(max(p)) == q.index(max(q)))
                    delta.extend(abs(x - y) for x, y in zip(p, q))
            ms.append(r["latency_ms"])
            return r["answers"]

        def score(p, target, key):
            top = max(range(len(p)), key=p.__getitem__)
            hits[key] += top == target
            preds.append((p[top], top == target, log_loss(p, target), brier(p, target)))

        for it in sev:
            a = ask(it["text"], {
                "severity": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": LEVELS},
                "blocked": {"type": "noul", "instructions": "Is the user completely blocked from using the product?"},
            })
            score(probs(a["severity"]), it["level"], "severity")
            score(probs(a["blocked"]), int(it["level"] == 2), "blocked")
        for it in route:
            a = ask(it["text"], {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": TEAMS}})
            score(probs(a["team"]), list(TEAMS).index(it["team"]), "route")

        n = {"severity": len(sev), "blocked": len(sev), "route": len(route)}
        acc = "  ".join(f"{k}={hits[k] / n[k]:.2f}" for k in hits)
        total = len(preds)
        line = (f"{mode:>9}: {acc}   logloss {sum(p[2] for p in preds) / total:.3f}  brier {sum(p[3] for p in preds) / total:.3f}"
                f"  ece {ece([(p[0], p[1]) for p in preds], args.bins):.3f}   median {sorted(ms)[len(ms) // 2]:.0f} ms")
        if args.ref:
            line += f"   vs ref: same answer {sum(same)}/{len(same)}, |dp| mean {sum(delta) / len(delta):.4f} max {max(delta):.3f}"
        print(line)


if __name__ == "__main__":
    main()
