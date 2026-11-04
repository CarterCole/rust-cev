"""Reference answers for a Kev checkpoint, computed from its published parts
without the Kev package: the Qwen3.5 base with the LoRA adapter merged (PEFT),
Kev's record layout (special tokens delimit state, question, options and the
decision slot) and the pointer head from `head.pt`, all in f32 on the CPU.

    uv run python parity/dump_kev.py jaredpalmer/kev-0.8b out.json
"""
import json, re, sys
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

repo, out = sys.argv[1], sys.argv[2]
adapter = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "head.pt"])
cfg = json.load(open(f"{adapter}/adapter_config.json"))
head = torch.load(f"{adapter}/head.pt", map_location="cpu", weights_only=True)
base_id, revision = cfg["base_model_name_or_path"], head.get("base_revision")
tok = AutoTokenizer.from_pretrained(base_id, revision=revision)
lm = AutoModelForCausalLM.from_pretrained(base_id, revision=revision, dtype=torch.float32, attn_implementation="eager").model
lm = PeftModel.from_pretrained(lm, adapter).merge_and_unload().eval()
T = float(head.get("temperature", 1.0))
q_w, q_b, k_w, k_b = (head["head"][k].float() for k in ("q.weight", "q.bias", "k.weight", "k.bias"))
dp = q_w.shape[0]

STATE, Q, OPT, OPT_END, DECIDE = "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"
esc = lambda s: re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", s)


def render(v, indent=0):
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, bool): return str(v)
    if isinstance(v, (str, int, float)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


def options(q):
    c = q.get("criteria")
    if q["type"] == "noul":
        c = c or {}
        return [n if not d else f"{n}: {render(d)}" for n, d in (("no", c.get("false")), ("yes", c.get("true")))]
    if q["type"] == "choice":
        return [k if v in (None, "") else f"{k}: {render(v)}" for k, v in c.items()]
    return [render(x) for x in c]


requests = [
    {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
     "questions": {
         "department": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}},
         "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
         "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}}},
    {"state": {"subject": "Login broken", "body": "I cannot sign in since the update; the app crashes on launch.", "tier": "pro"},
     "questions": {
         "team": {"type": "choice", "instructions": "Route the ticket", "criteria": {"billing": None, "tech": None, "sales": None}},
         "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": {"true": "money back is requested", "false": "no refund mentioned"}}}},
]
rows = []
with torch.no_grad():
    for r in requests:
        state_ids = tok.encode(STATE + esc(render(r["state"])), add_special_tokens=False)
        answers = {}
        for name, q in r["questions"].items():
            branch = Q + esc(q["instructions"]) + "".join(OPT + esc(o) + OPT_END for o in options(q)) + DECIDE
            ids = state_ids + tok.encode(branch, add_special_tokens=False)
            h = lm(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0]
            opt_pos = [i for i, t in enumerate(ids) if t == tok.convert_tokens_to_ids(OPT_END)]
            hd, ho = h[-1], h[opt_pos]
            z = ((ho @ k_w.T + k_b) @ (hd @ q_w.T + q_b)) / dp ** 0.5 / T
            answers[name] = {"probs": torch.softmax(z, -1).tolist(), "tokens": len(ids)}
        rows.append({"request": r, "answers": answers})
json.dump({"repo": repo, "temperature": T, "rows": rows}, open(out, "w"), indent=1)
print("wrote", out)
for r in rows:
    for n, a in r["answers"].items():
        print(n, [round(p, 4) for p in a["probs"]])
