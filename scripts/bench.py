#!/usr/bin/env python3
"""Latency benchmark against a running cev server (stdlib only).

    python3 scripts/bench.py http://127.0.0.1:8080

Reports median latency for a short and a ~2k-token state, with the state
prefix cached (same state repeated) and uncached (state changes every call).
"""

import copy
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "Payments, invoices, refunds", "tech": "Bugs, crashes, errors", "sales": "New purchases and upgrades"}},
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "severity": {"type": "score", "instructions": "How severe is the issue?",
                 "criteria": ["Cosmetic; no impact", "Degraded, workaround exists", "Blocking; no workaround"]},
}
TICKET = {"subject": "App crashes on login", "body": "Since the update, the iOS app crashes as soon as I tap Log in."}
HISTORY = [f"Earlier message {i}: the customer described several steps they tried, including reinstalling the app and clearing the cache." for i in range(80)]


def post(body):
    req = urllib.request.Request(BASE + "/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req).read())
    return (time.perf_counter() - t) * 1e3, r["usage"]["input_tokens"]


def median(xs):
    return sorted(xs)[len(xs) // 2]


def run(state, n=7):
    body = {"state": state, "questions": QUESTIONS, "no_store": True}
    post(body)
    cached = [post(body) for _ in range(n)]
    uncached = []
    for i in range(n):
        b = copy.deepcopy(body)
        b["state"]["nonce"] = f"run-{time.time()}-{i}"
        uncached.append(post(b)[0])
    return median([c[0] for c in cached]), median(uncached), cached[0][1]


for name, state in [("short", {"ticket": TICKET}), ("long", {"ticket": TICKET, "history": HISTORY})]:
    c, u, tok = run(state)
    print(f"{name:>5}: {tok:5d} tokens, 3 questions | prefix cached {c:6.1f} ms | uncached {u:6.1f} ms")
