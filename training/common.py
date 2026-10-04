"""Shared helpers: read cev export rows and score answer codes with HF models.

Export rows (GET /v1/export) carry the exact prompt text cev served, so
training here sees byte-identical inputs to serving. The answer is the next
token after `prompt`, restricted to `codes`.
"""

import json
import urllib.request
from pathlib import Path

import torch


def load_rows(source: str, labeled: bool = True) -> list[dict]:
    """`source` is a .jsonl file or a cev server URL (reads /v1/export)."""
    if source.startswith("http"):
        url = source.rstrip("/") + f"/v1/export?labeled={'true' if labeled else 'false'}"
        with urllib.request.urlopen(url) as r:
            text = r.read().decode()
    else:
        text = Path(source).read_text()
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    return [r for r in rows if r.get("target")] if labeled else rows


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def code_ids(tokenizer, codes: list[str]) -> list[int]:
    ids = []
    for c in codes:
        t = tokenizer.encode(c, add_special_tokens=False)
        assert len(t) == 1, f"code {c!r} is not a single token"
        ids.append(t[0])
    return ids


def code_logits(model, tokenizer, row: dict, device) -> torch.Tensor:
    """Logits of the row's answer codes at the answer position, shape (k,)."""
    ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, logits_to_keep=1) if _supports_keep(model) else model(input_ids=x)
    last = out.logits[0, -1]
    return last[torch.tensor(code_ids(tokenizer, row["codes"]), device=device)].float()


def _supports_keep(model) -> bool:
    try:
        import inspect

        return "logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
