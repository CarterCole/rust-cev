"""Check that cev's Rust forward pass matches Hugging Face transformers.

    uv run python parity.py --model Qwen/Qwen3-0.6B --source http://127.0.0.1:8080

Recomputes the base option distribution of recent decisions with transformers
(float32) and compares it to what cev computed.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import code_logits, load_rows, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="cev server URL or export .jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    rows = load_rows(args.source, labeled=False)[-args.limit :]
    worst, agree = 0.0, 0
    with torch.no_grad():
        for r in rows:
            p_hf = torch.softmax(code_logits(model, tok, r, device), -1).cpu()
            p_cev = torch.tensor(r["base_probabilities"])
            d = (p_hf - p_cev).abs().max().item()
            worst = max(worst, d)
            agree += int(p_hf.argmax() == p_cev.argmax())
            print(f"{r['question_id']:>10} {r['type']:>6}  max|Δp|={d:.4f}  hf={p_hf.numpy().round(3)} cev={p_cev.numpy().round(3)}")
    print(f"\n{len(rows)} decisions: argmax agreement {agree}/{len(rows)}, worst max|Δp| {worst:.4f}")


if __name__ == "__main__":
    main()
