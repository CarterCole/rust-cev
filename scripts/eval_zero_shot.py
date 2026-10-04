#!/usr/bin/env python3
"""Zero-shot accuracy of a running cev server on small built-in tasks, per
debias mode (stdlib only).

    python3 scripts/eval_zero_shot.py http://127.0.0.1:8080 [REFERENCE_URL]

With a second server (e.g. the f32 model next to a quantized one) it also
reports how closely the first tracks it: how often they pick the same answer
and the mean and largest difference in any probability.
"""

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
REF = sys.argv[2] if len(sys.argv) > 2 else None
src = open(__file__.replace("eval_zero_shot.py", "demo_online.py")).read()
TICKETS = eval(src[src.index("TICKETS = [") + 10 : src.index("]\nTRAIN") + 1])
LEVELS = ["Cosmetic; no impact", "Degraded, workaround exists", "Blocking; no workaround"]
ROUTE = [
    ("I was charged twice for my subscription, please refund.", "billing"), ("My invoice shows the wrong VAT number.", "billing"),
    ("Can I get a receipt for last month's payment?", "billing"), ("Cancel my plan and refund the unused days.", "billing"),
    ("The app crashes when I open settings.", "tech"), ("Error 500 when saving a document.", "tech"),
    ("Sync stopped working after the update.", "tech"), ("The export button does nothing.", "tech"),
    ("How much is the enterprise plan for 200 seats?", "sales"), ("Do you offer discounts for nonprofits?", "sales"),
    ("I'd like to upgrade to the Pro plan.", "sales"), ("Can we get a demo for our sales team?", "sales"),
]
TEAMS = {"billing": "Payments, invoices, refunds", "tech": "Bugs, crashes, errors", "sales": "Pricing, new purchases, upgrades"}


def post(base, body):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def probs(a):
    return [a["noul"], 1 - a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())


def ask(state, questions, debias):
    body = {"state": state, "questions": questions, "no_store": True, "debias": debias}
    r = post(BASE, body)
    if REF:
        for qid, a in post(REF, body)["answers"].items():
            p, q = probs(r["answers"][qid]), probs(a)
            same.append(p.index(max(p)) == q.index(max(q)))
            delta.extend(abs(x - y) for x, y in zip(p, q))
    return r["answers"], r["latency_ms"]


for mode in ["none", "calibrate", "permute", "full"]:
    hits = {"severity": 0, "blocked": 0, "route": 0}
    ms, same, delta = [], [], []
    for text, level in TICKETS:
        a, t = ask(text, {
            "severity": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": LEVELS},
            "blocked": {"type": "noul", "instructions": "Is the user completely blocked from using the product?"},
        }, mode)
        ms.append(t)
        p = a["severity"]["probabilities"]
        hits["severity"] += max(p, key=p.get) == str(level)
        hits["blocked"] += (a["blocked"]["noul"] >= 0.5) == (level == 2)
    for text, team in ROUTE:
        a, t = ask(text, {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": TEAMS}}, mode)
        hits["route"] += a["team"]["choice"] == team
    n = {"severity": len(TICKETS), "blocked": len(TICKETS), "route": len(ROUTE)}
    print(f"{mode:>9}: " + "  ".join(f"{k}={hits[k] / n[k]:.2f}" for k in hits) + f"   median {sorted(ms)[len(ms) // 2]:.0f} ms (2 questions)"
          + (f"   vs ref: same answer {sum(same)}/{len(same)}, |dp| mean {sum(delta) / len(delta):.4f} max {max(delta):.3f}" if REF else ""))
