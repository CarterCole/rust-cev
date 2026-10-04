#!/usr/bin/env python3
"""Online-learning demo against a running cev server (stdlib only).

    cev --model Qwen/Qwen3-0.6B --min-examples 4 &
    python3 scripts/demo_online.py http://127.0.0.1:8080

Teaches a severity scale with labelled tickets via /v1/examples, then compares
held-out accuracy of the base model (the prequential "base" answer) with the
adapted answer. Resets the task first so it can be re-run.
"""

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
TASK = "demo_severity"
LEVELS = ["Cosmetic; no impact", "Degraded, workaround exists", "Blocking; no workaround"]

# (ticket, level)
TICKETS = [
    ("The logo on the invoice PDF is slightly blurry.", 0),
    ("Typo on the pricing page: 'recieve' should be 'receive'.", 0),
    ("The settings icon is misaligned by a few pixels on Android.", 0),
    ("Dark mode makes the footer text a little hard to read.", 0),
    ("The confirmation email uses the old brand color.", 0),
    ("Tooltip on the export button has a spelling mistake.", 0),
    ("Profile avatars look stretched on the team page.", 0),
    ("The loading spinner is off-center on Safari.", 0),
    ("Button labels are in title case instead of sentence case.", 0),
    ("The date in the footer shows last year.", 0),
    ("CSV export is slow but finishes after a minute or two.", 1),
    ("Search sometimes misses results; filtering by date manually works.", 1),
    ("Push notifications are delayed by about 10 minutes.", 1),
    ("I can't upload images by drag and drop, but the upload button works.", 1),
    ("The dashboard charts fail to load on Firefox; Chrome is fine.", 1),
    ("Password reset emails take a while; retrying eventually works.", 1),
    ("Keyboard shortcuts stopped working, I use the menu instead.", 1),
    ("Autosave fails sometimes, so I save manually every few minutes.", 1),
    ("The mobile app logs me out daily; logging back in works.", 1),
    ("Bulk edit is broken, but editing items one by one still works.", 1),
    ("The app crashes immediately on launch for every user in our org.", 2),
    ("Checkout returns a 500 error; no customer can pay right now.", 2),
    ("All our data disappeared from the dashboard after the update.", 2),
    ("Login is down for everyone; we are completely locked out.", 2),
    ("The API returns errors for every request; our integration is dead.", 2),
    ("Payments are being charged twice and we can't stop it.", 2),
    ("The server won't start after the upgrade; production is down.", 2),
    ("Files uploaded yesterday are corrupted and can't be opened at all.", 2),
    ("Nobody can send messages; the send button does nothing on any device.", 2),
    ("The database migration failed and the app shows a blank page.", 2),
]
TRAIN = [t for i, t in enumerate(TICKETS) if i % 10 < 7]
TEST = [t for i, t in enumerate(TICKETS) if i % 10 >= 7]


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def question():
    return {"sev": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": LEVELS, "task": TASK}}


def predict(text):
    a = call("POST", "/v1/systemone", {"state": text, "questions": question(), "no_store": True})["answers"]["sev"]
    probs = [a["probabilities"][str(i)] for i in range(3)]
    base = a.get("x_cev", {}).get("base_probabilities") or probs
    return max(range(3), key=probs.__getitem__), max(range(3), key=base.__getitem__)


def accuracy(rows, idx):
    preds = [predict(t) for t, _ in rows]
    return sum(p[idx] == y for p, (_, y) in zip(preds, rows)) / len(rows)


try:
    call("DELETE", f"/v1/tasks/{TASK}")
except Exception:
    pass

print(f"base model, held-out accuracy: {accuracy(TEST, 1):.2f}")
order = sorted(TRAIN, key=lambda t: hash(t[0]) % 97)  # interleave levels
examples = [{"state": t, "questions": question(), "labels": {"sev": y}} for t, y in order]
res = call("POST", "/v1/examples", {"examples": examples})["results"]
losses = [r["feedback"][0]["served_loss"] for r in res]
print(f"taught {len(res)} labels; served loss first half {sum(losses[:len(losses)//2])/(len(losses)//2):.2f}, "
      f"second half {sum(losses[len(losses)//2:])/(len(losses)-len(losses)//2):.2f}")
info = call("GET", f"/v1/tasks/{TASK}")
print(f"adapter: active={info['active']} T={info['temperature']:.2f} "
      f"prequential loss base={info['base_loss']:.3f} adapted={info['adapted_loss']:.3f}")
print(f"adapted,    held-out accuracy: {accuracy(TEST, 0):.2f}")
