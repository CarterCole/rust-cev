"""Fine-tune the backbone on cev feedback, then merge into weights cev can serve.

    # labels straight from a running server (or a saved export .jsonl)
    uv run python train_lora.py --base Qwen/Qwen3-1.7B --source http://127.0.0.1:8080 --out ./ckpt/v1
    cev --model ./ckpt/v1

Loss is soft cross-entropy over the answer-code logits at the answer position
only (no other vocabulary token is supervised), weighted per row. The prompt
text comes from cev's export, so training and serving inputs are identical.
A deterministic 10% of decisions is held out for before/after evaluation.
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import code_logits, load_rows, pick_device

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def held_out(row, frac):
    h = int(hashlib.sha256(row["decision_id"].encode()).hexdigest()[:8], 16)
    return (h % 1000) < frac * 1000


def loss_of(model, tok, row, device):
    z = code_logits(model, tok, row, device)
    t = torch.tensor(row["target"], device=device)
    return -(t * torch.log_softmax(z, -1)).sum() * float(row.get("weight", 1.0)), z


@torch.no_grad()
def evaluate(model, tok, rows, device):
    if not rows:
        return float("nan"), float("nan")
    model.eval()
    tot, hit = 0.0, 0
    for r in rows:
        l, z = loss_of(model, tok, r, device)
        tot += l.item()
        hit += int(z.argmax().item() == max(range(len(r["target"])), key=r["target"].__getitem__))
    return tot / len(rows), hit / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF repo id or directory of the backbone cev serves")
    ap.add_argument("--source", required=True, help="cev server URL or export .jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default=None, help="only train on this task")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    rows = load_rows(args.source, labeled=True)
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
    tok = AutoTokenizer.from_pretrained(args.base)
    rows = [r for r in rows if len(tok.encode(r["prompt"], add_special_tokens=False)) <= args.max_tokens]
    train = [r for r in rows if not held_out(r, args.holdout)]
    test = [r for r in rows if held_out(r, args.holdout)]
    if not train:
        raise SystemExit("no labelled rows to train on")
    print(f"{len(train)} train / {len(test)} held-out rows on {device}")

    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=TARGETS))
    model.print_trainable_parameters()

    before = evaluate(model, tok, test, device)
    print(f"held-out before: loss {before[0]:.3f} acc {before[1]:.3f}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * max(0.0, 1 - s / steps))
    step = 0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)  # option order is fixed by the prompt; shuffle rows only
        running = 0.0
        for i, r in enumerate(train):
            loss, _ = loss_of(model, tok, r, device)
            (loss / args.accum).backward()
            running += loss.item()
            step += 1
            if step % args.accum == 0 or i == len(train) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            sched.step()
        print(f"epoch {epoch + 1}: train loss {running / len(train):.3f}")
    after = evaluate(model, tok, test, device)
    print(f"held-out after:  loss {after[0]:.3f} acc {after[1]:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    # Keep tokenizer_config's chat template flag so cev picks the same prompt framing.
    base_dir = Path(args.base)
    if base_dir.is_dir() and (base_dir / "tokenizer_config.json").exists():
        shutil.copy(base_dir / "tokenizer_config.json", out / "tokenizer_config.json")
    (out / "cev_training.json").write_text(json.dumps({
        "base": args.base, "rows": len(train), "held_out": len(test), "epochs": args.epochs, "lr": args.lr,
        "rank": args.rank, "before": before, "after": after,
    }, indent=2))
    print(f"saved merged model to {out}; serve it with: cev --model {out}")


if __name__ == "__main__":
    main()
