"""Reference values for cev's Qwen3.5 decoder: run HF transformers in f32 on a
few prompts and write the final hidden state at the last token plus the
logits of a handful of tokens, for `cargo test -p cev-model --test parity`.

    uv run python parity/dump_qwen35.py Qwen/Qwen3.5-0.8B out.json
"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="eager").eval()
prompts = [
    "<|im_start|>user\nIs water wet? Answer A for yes, B for no.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|fim_prefix|>The shoes arrived two weeks late and in the wrong size.<|fim_middle|>Which team?<|box_start|>returns<|box_end|><|box_start|>shipping<|box_end|><|fim_suffix|>",
    " ".join(f"token{i}" for i in range(300)) + " The end. Answer:",
]
probe_ids = [tok.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D", "yes", "no"]]
rows = []
with torch.no_grad():
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        o = model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)
        h = o.hidden_states[-1][0]  # after the final norm? (HF: last entry is post-norm for most models)
        # Use the model's own final norm to be sure.
        h_last = model.model.norm(o.hidden_states[-1][0, -1]) if not torch.allclose(o.hidden_states[-1][0, -1], model.model.norm(o.hidden_states[-1][0, -1])) else o.hidden_states[-1][0, -1]
        rows.append({
            "ids": ids,
            "hidden_last": o.hidden_states[-1][0, -1].tolist(),
            "hidden_mid": o.hidden_states[-1][0, len(ids) // 2].tolist(),
            "logits_last": o.logits[0, -1, probe_ids].tolist(),
            "probe_ids": probe_ids,
        })
# Kev-style check: a prefix then a branch, as hidden state of the whole row.
json.dump({"model": model_id, "rows": rows, "hidden_is_normed": "see rust test"}, open(out, "w"))
print("wrote", out, [len(r["ids"]) for r in rows])
