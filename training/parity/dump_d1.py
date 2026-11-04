"""Reference answers for Liquid's d1 (LFM2.5-VL), computed with HF's stock
`Lfm2VlForConditionalGeneration` and d1's prompt contract (`prompt.py`):
the state and QUESTION block, option codes / yes-no / digits read from the
last-position log-probabilities (max over spellings), in f32 on the CPU.

    uv run python parity/dump_d1.py LiquidAI/d1-3B out.json
"""
import json, sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

repo, out = sys.argv[1], sys.argv[2]
processor = AutoProcessor.from_pretrained(repo)
tok = processor.tokenizer
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager").eval()
bos = tok.bos_token

FALLBACK = [chr(c) for c in range(65, 91)] + [f"{i:02d}" for i in range(100)] + [chr(c) for c in range(97, 123)] + [f"#{i}" for i in range(200)] + [chr(a) + chr(b) for a in range(65, 91) for b in range(65, 91)]


def option_codes(labels):
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return [chr(65 + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


def aliases(labels):
    used, outp = set(), []
    def take(raw):
        enc = tok.encode(raw, add_special_tokens=False)
        if len(enc) != 1 or enc[0] in used:
            return False
        outp.append((raw, enc[0])); used.add(enc[0]); return True
    for code in option_codes(labels):
        if take(code):
            continue
        if not any(take(raw) for raw in FALLBACK):
            raise RuntimeError("no alias")
    return outp


def ids(texts):
    o, seen = [], set()
    for t in texts:
        e = tok.encode(t, add_special_tokens=False)
        if len(e) == 1 and e[0] not in seen:
            o.append(e[0]); seen.add(e[0])
    return o


def readout_ids(q):
    if q["type"] == "noul":
        return [ids(["yes", "Yes", "YES"]), ids(["no", "No", "NO"])]
    if q["type"] == "score":
        return [ids([str(i)]) for i in range(len(q["criteria"]))]
    groups = []
    for code, tid in aliases(list(q["criteria"])):
        extra = ids([f" {code}"])
        groups.append([tid] + [i for i in extra if i != tid])
    return groups


def question_block(q):
    if q["type"] == "choice":
        labels = list(q["criteria"])
        codes = aliases(labels)
        lines = "\n".join(f"{codes[i][0]} {q['criteria'][lab] or lab.replace('_', ' ')}" for i, lab in enumerate(labels))
        return f"{q['instructions']}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if q["type"] == "noul":
        extra = ""
        if q.get("criteria"):
            extra = f"\nYes: {q['criteria'].get('true')}\nNo: {q['criteria'].get('false')}"
        return f"{q['instructions']}{extra}\n\nReply with yes or no only."
    legend = "\n".join(f"{i} {name}" for i, name in enumerate(q["criteria"]))
    return f"{q['instructions']}\n\n{legend}\n\nReply with a single digit 0-{len(q['criteria']) - 1} only."


def state_block(state):
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def prefix_text(state, images=""):
    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    return f"{bos}<|im_start|>user\n{images}{body}"


def suffix_text(q):
    return f"{question_block(q)}<|im_end|>\n<|im_start|>assistant\n"


def readout(q, logits):
    logz = logits - torch.logsumexp(logits, -1)
    scores = torch.tensor([max(float(logz[i]) for i in g) for g in readout_ids(q)])
    probs = torch.softmax(scores, -1).tolist()
    # d1 reads a noul as [yes, no]; cev (and these files) use [false, true].
    return probs[::-1] if q["type"] == "noul" else probs


h, w = 160, 224
yy, xx = np.mgrid[0:h, 0:w]
rgb = np.stack([xx * 255 // (w - 1), yy * 255 // (h - 1), ((xx + yy) * 255 // (w + h - 2))], -1).astype(np.uint8)
img = Image.fromarray(rgb)
img.save(out.replace(".json", ".png"))

requests = [
    {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
     "questions": {
         "department": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}},
         "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
         "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}}},
    {"state": {"subject": "Login broken", "body": "I cannot sign in since the update; the app crashes on launch."},
     "questions": {"team": {"type": "choice", "instructions": "Route the ticket", "criteria": {"billing": None, "tech": None, "sales": None}},
                   "refund": {"type": "noul", "instructions": "Refund requested?", "criteria": {"true": "money back", "false": "no refund"}}}},
    {"state": None, "images": True,
     "questions": {"corner": {"type": "choice", "instructions": "What is the dominant colour at the top-left corner?",
                              "criteria": {"red": None, "green": None, "blue": None, "black": None}},
                   "photo": {"type": "noul", "instructions": "Is this a gradient image?"}}},
]
rows = []
with torch.no_grad():
    for r in requests:
        answers = {}
        for name, q in r["questions"].items():
            if r.get("images"):
                text = prefix_text(r["state"], "<image>") + suffix_text(q)
                inputs = processor(text=[text], images=[[img]], return_tensors="pt", add_special_tokens=False)
                n = int(inputs["pixel_attention_mask"].sum(1).max())
                inputs["pixel_values"] = inputs["pixel_values"][:, :n]
                inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"][:, :n]
                logits = model(**inputs, logits_to_keep=1).logits[0, -1]
                ntok = int(inputs["input_ids"].shape[1])
                prompt = tok.decode(inputs["input_ids"][0])
            else:
                text = prefix_text(r["state"]) + suffix_text(q)
                ids_ = tok.encode(text, add_special_tokens=False)
                logits = model(input_ids=torch.tensor([ids_]), logits_to_keep=1).logits[0, -1]
                ntok = len(ids_)
                prompt = text
            answers[name] = {"probs": readout(q, logits), "tokens": ntok, "prompt": prompt}
        req = {k: v for k, v in r.items() if k != "images"}
        if r.get("images"):
            import base64, io
            buf = io.BytesIO(); img.save(buf, format="PNG")
            req["images"] = [base64.b64encode(buf.getvalue()).decode()]
        rows.append({"request": req, "answers": answers})
json.dump({"repo": repo, "rows": rows}, open(out, "w"), indent=1)
print("wrote", out)
for r in rows:
    for n, a in r["answers"].items():
        print(n, [round(p, 4) for p in a["probs"]], a["tokens"])
