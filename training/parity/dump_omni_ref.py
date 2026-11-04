"""Reference answers for d1-omni from its weights alone: this file re-states the
model's math (bidirectional LFM2.5 trunk with a centred 3-tap convolution,
the two-layer decision head over `<|mask|>` markers, a SigLIP2 image path and
the FastConformer speech path) with plain torch modules named like the
checkpoint's tensors, so `load_state_dict` is the only coupling. Text, image
and audio requests, f32 on the CPU.

    uv run python parity/dump_omni_ref.py LiquidAI/d1-omni-600M out.json
"""
import base64, io, json, math, re, sys, wave
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import Siglip2VisionConfig, Siglip2VisionModel

repo, out = sys.argv[1], sys.argv[2]
root = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])
cfg = json.load(open(f"{root}/config.json"))
tok = Tokenizer.from_file(f"{root}/tokenizer.json")
tc, vc, ac = cfg["text_config"], cfg["vision_config"], cfg["audio_config"]
D = tc["hidden_size"]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads, self.kv = tc["num_attention_heads"], tc["num_key_value_heads"]; self.hd = D // self.heads
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(D, D, bias=False), nn.Linear(D, self.kv * self.hd, bias=False), nn.Linear(D, self.kv * self.hd, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)
        self.q_layernorm, self.k_layernorm = RMSNorm(self.hd, tc["norm_eps"]), RMSNorm(self.hd, tc["norm_eps"])
    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        rot = lambda z: z * cos + torch.cat((-z[..., z.shape[-1] // 2:], z[..., : z.shape[-1] // 2]), -1) * sin
        q = rot(self.q_layernorm(self.q_proj(x).view(b, t, -1, self.hd)).transpose(1, 2))
        k = rot(self.k_layernorm(self.k_proj(x).view(b, t, -1, self.hd)).transpose(1, 2))
        v = self.v_proj(x).view(b, t, -1, self.hd).transpose(1, 2)
        g = self.heads // self.kv
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.hd ** -0.5)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, -1))


class ShortConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(D, D, 3, groups=D, bias=False, padding=2)
        self.in_proj, self.out_proj = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
    def forward(self, x, pad, keep_right):
        b, c, u = self.in_proj(x * pad[:, :, None]).transpose(-1, -2).chunk(3, dim=-2)
        bx = b * u; w = self.conv.weight[:, 0, :]; t = bx.shape[-1]
        xp = F.pad(bx, (1, 1))
        y = xp[..., 0:t] * w[:, 0, None] + xp[..., 1:1 + t] * w[:, 1, None] + xp[..., 2:2 + t] * keep_right[:, None, :] * w[:, 2, None]
        return self.out_proj((c * y).transpose(-1, -2))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = int(2 * tc["intermediate_size"] / 3); hidden = int(tc["block_ffn_dim_multiplier"] * hidden)
        hidden = tc["block_multiple_of"] * ((hidden + tc["block_multiple_of"] - 1) // tc["block_multiple_of"])
        self.w1, self.w3, self.w2 = nn.Linear(D, hidden, bias=False), nn.Linear(D, hidden, bias=False), nn.Linear(hidden, D, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.attn = kind == "full_attention"
        if self.attn: self.self_attn = Attention()
        else: self.conv = ShortConv()
        self.feed_forward = MLP(); self.operator_norm = RMSNorm(D, tc["norm_eps"]); self.ffn_norm = RMSNorm(D, tc["norm_eps"])


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(tc["vocab_size"], D)
        self.layers = nn.ModuleList(Layer(k) for k in tc["layer_types"])
        self.embedding_norm = RMSNorm(D, tc["norm_eps"])
    def forward(self, h, pad, prefix):
        b, t, _ = h.shape
        hd = D // tc["num_attention_heads"]
        inv = 1.0 / (tc["rope_theta"] ** (torch.arange(0, hd, 2).float() / hd))
        pos = torch.arange(t).float()
        fr = torch.outer(pos, inv); emb = torch.cat((fr, fr), -1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        ar = torch.arange(t)[None]
        media_q, text_k = ar < prefix[:, None], ar >= prefix[:, None]
        mask = torch.zeros(b, 1, t, t) + ((~pad).float() * -1e9)[:, None, None, :]
        mask = mask.masked_fill((media_q[:, :, None] & text_k[:, None, :])[:, None], -1e9)
        keep_right = (ar != prefix[:, None] - 1).float()
        for l in self.layers:
            x = l.operator_norm(h)
            x = l.self_attn(x, cos, sin, mask) if l.attn else l.conv(x, pad.float(), keep_right)
            h = h + x
            h = h + l.feed_forward(l.ffn_norm(h))
        return self.embedding_norm(h)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(3, D)
        layer = nn.TransformerEncoderLayer(D, D // 64, 4 * D, 0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, cfg["head_layers"], enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
    def forward(self, h, pad, markers, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        for l in self.head.layers:
            h = l(h, src_key_padding_mask=~pad)
        return [self.scorer(h[i, m]).squeeze(-1) for i, m in enumerate(markers)]


class Projector(nn.Module):
    def __init__(self, vd, hidden, outd):
        super().__init__(); self.linear_1, self.linear_2 = nn.Linear(vd * 4, hidden), nn.Linear(hidden, outd)
    def forward(self, x):
        b, h, w, c = x.shape
        x = x.reshape(b, h, w // 2, c * 2).permute(0, 2, 1, 3).reshape(b, w // 2, h // 2, c * 4).permute(0, 2, 1, 3)
        return self.linear_2(F.gelu(self.linear_1(x))).reshape(b, -1, self.linear_2.out_features)


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        c = Siglip2VisionConfig(**vc); c._attn_implementation = "eager"
        self.tower = nn.Module(); self.tower.vision_model = Siglip2VisionModel(c)
        self.projector = Projector(vc["hidden_size"], cfg["projector_hidden_size"], D)


# ---- audio (FastConformer) ----
class FeedForward(nn.Module):
    def __init__(self, d, ff):
        super().__init__(); self.linear1, self.linear2 = nn.Linear(d, ff), nn.Linear(ff, d)
    def forward(self, x): return self.linear2(F.silu(self.linear1(x)))


class RelAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h, self.dk = h, d // h
        self.linear_q, self.linear_k, self.linear_v, self.linear_out = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.linear_pos = nn.Linear(d, d, bias=False)
        self.pos_bias_u, self.pos_bias_v = nn.Parameter(torch.zeros(h, self.dk)), nn.Parameter(torch.zeros(h, self.dk))
    def forward(self, x, pos):
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, self.h, self.dk); k = self.linear_k(x).view(b, t, self.h, self.dk).transpose(1, 2); v = self.linear_v(x).view(b, t, self.h, self.dk).transpose(1, 2)
        p = self.linear_pos(pos).view(1, -1, self.h, self.dk).transpose(1, 2)
        ac = (q + self.pos_bias_u).transpose(1, 2) @ k.transpose(-2, -1)
        bd = (q + self.pos_bias_v).transpose(1, 2) @ p.transpose(-2, -1)
        bh, hh, ql, pl = bd.shape
        bd = F.pad(bd, (1, 0)).view(bh, hh, pl + 1, ql)[:, :, 1:].view(bh, hh, ql, pl)
        s = (ac + bd[:, :, :, :t]) / math.sqrt(self.dk)
        return self.linear_out((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(b, t, -1))


class ConvMod(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.pointwise_conv1, self.depthwise_conv, self.batch_norm, self.pointwise_conv2 = nn.Conv1d(d, 2 * d, 1), nn.Conv1d(d, d, k, groups=d), nn.BatchNorm1d(d), nn.Conv1d(d, d, 1)
        self.pad = (k - 1) // 2
    def forward(self, x):
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        x = self.depthwise_conv(F.pad(x, (self.pad, self.pad)))
        return self.pointwise_conv2(F.silu(self.batch_norm(x))).transpose(1, 2)


class ConfLayer(nn.Module):
    def __init__(self, d, ff, h, k):
        super().__init__()
        self.norm_feed_forward1, self.feed_forward1 = nn.LayerNorm(d), FeedForward(d, ff)
        self.norm_self_att, self.self_attn = nn.LayerNorm(d), RelAttn(d, h)
        self.norm_conv, self.conv = nn.LayerNorm(d), ConvMod(d, k)
        self.norm_feed_forward2, self.feed_forward2 = nn.LayerNorm(d), FeedForward(d, ff)
        self.norm_out = nn.LayerNorm(d)
    def forward(self, x, pos):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Sub(nn.Module):
    def __init__(self, fi, ch, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, ch, 3, 2, 1), nn.ReLU(True), nn.Conv2d(ch, ch, 3, 2, 1, groups=ch), nn.Conv2d(ch, ch, 1), nn.ReLU(True), nn.Conv2d(ch, ch, 3, 2, 1, groups=ch), nn.Conv2d(ch, ch, 1), nn.ReLU(True))
        f = fi
        for _ in range(3): f = (f - 1) // 2 + 1
        self.out = nn.Linear(ch * f, d)
    def forward(self, x, n):
        x = self.conv(x.unsqueeze(1)); b, c, t, f = x.shape
        for _ in range(3): n = (n + 2 - 3) // 2 + 1
        return self.out(x.transpose(1, 2).reshape(b, t, c * f)), n


class Conformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = ac["d_model"]; self.d = d
        self.pre_encode = Sub(ac["feat_in"], ac["subsampling_conv_channels"], d)
        self.layers = nn.ModuleList(ConfLayer(d, d * ac["ff_expansion_factor"], ac["n_heads"], ac["conv_kernel_size"]) for _ in range(ac["n_layers"]))
    def forward(self, mel, n):
        x, n = self.pre_encode(mel.transpose(1, 2), n)
        x = x[:, :n]
        t = x.shape[1]
        positions = torch.arange(t - 1, -t, -1, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.d, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d))
        pe = torch.zeros(len(positions), self.d); pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * div), torch.cos(positions * div)
        for l in self.layers: x = l(x, pe[None])
        return x


class Adapter(nn.Module):
    def __init__(self, di, do):
        super().__init__(); self.norm, self.linear_1, self.linear_2 = nn.LayerNorm(di), nn.Linear(di, do), nn.Linear(do, do)
    def forward(self, x): return self.linear_2(F.gelu(self.linear_1(self.norm(x))))


class Residual(nn.Module):
    def __init__(self, d, w):
        super().__init__(); self.ln, self.down, self.up = nn.LayerNorm(d), nn.Linear(d, w), nn.Linear(w, d)
    def forward(self, x): return x + self.up(F.gelu(self.down(self.ln(x))))


class Audio(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = Conformer(); self.adapter = Adapter(ac["d_model"], D); self.residual = Residual(D, ac["residual_width"])


def slaney(sr=16000, n_fft=512, n_mels=128):
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    def hz_to_mel(f): return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp
    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64); f = f_sp * m; high = m >= min_log_mel
        f[high] = min_log_hz * np.exp(logstep * (m[high] - min_log_mel)); return f
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)); fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, np.fft.rfftfreq(n=n_fft, d=1.0 / sr))
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return torch.from_numpy(weights)


def mel_frontend(x):
    n = torch.tensor([x.shape[1]]); frames = (n + 512 - 512) // 160
    x = torch.cat((x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]), 1)
    win = torch.hann_window(400, periodic=False)
    s = torch.stft(x, n_fft=512, hop_length=160, win_length=400, center=True, window=win, return_complex=True, pad_mode="constant")
    p = torch.view_as_real(s).pow(2).sum(-1)
    m = torch.log(slaney()[None] @ p + 2 ** -24)
    valid = torch.arange(m.shape[2])[None] < frames[:, None]
    cnt = valid.sum(1)
    mean = torch.where(valid[:, None], m, 0.0).sum(2) / cnt[:, None]
    std = torch.sqrt(torch.where(valid[:, None], m - mean[:, :, None], 0.0).pow(2).sum(2) / (cnt[:, None] - 1.0))
    m = (m - mean[:, :, None]) / (std.masked_fill(std.isnan(), 0.0) + 1e-5)[:, :, None]
    return m.masked_fill(~valid[:, None], 0.0), frames


# ---- assemble ----
class Omni(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder, self.head, self.vision, self.audio = Trunk(), Head(), Vision(), Audio()


model = Omni()
sd = load_file(f"{root}/model.safetensors")
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [m for m in missing if "num_batches_tracked" not in m]
assert not missing and not unexpected, (missing[:5], unexpected[:5])
model.eval()

# ---- preprocessing (ports of the model's rules) ----
def layout(width, height):
    factor, maximum, minimum = 32, 256 * 1024, 64 * 1024
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum); h = max(factor, math.floor(height / beta / factor) * factor); w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width)); h = math.ceil(height * beta / factor) * factor; w = math.ceil(width * beta / factor) * factor
    large = max(16, round(height / factor) * factor) * max(16, round(width / factor) * factor) > maximum * 2
    grid = (1, 1)
    if large:
        ratios = sorted({(x, y) for n in range(2, 11) for x in range(1, n + 1) for y in range(1, n + 1) if 2 <= x * y <= 10}, key=lambda r: r[0] * r[1])
        best = float("inf")
        for ratio in ratios:
            diff = abs(width / height - ratio[0] / ratio[1])
            if diff < best or (diff == best and width * height > 0.5 * 512 * 512 * ratio[0] * ratio[1]): grid, best = ratio, diff
    return grid, (h, w), large


def image_prefix(image):
    from torchvision.transforms.v2 import functional as tvf
    (gw, gh), (h, w), large = layout(*image.size)
    x = tvf.pil_to_tensor(image.convert("RGB")); crops = []
    if large:
        big = tvf.resize(x, [gh * 512, gw * 512], interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)
        crops = [big[:, r * 512:(r + 1) * 512, c * 512:(c + 1) * 512] for r in range(gh) for c in range(gw)]
    crops.append(tvf.resize(x, [h, w], interpolation=tvf.InterpolationMode.BILINEAR, antialias=True))
    outs = []
    for crop in crops:
        crop = tvf.normalize(crop.float(), [127.5] * 3, [127.5] * 3); _, ch, cw = crop.shape; ph, pw = ch // 16, cw // 16
        patches = crop.reshape(3, ph, 16, pw, 16).permute(1, 3, 2, 4, 0).reshape(ph * pw, 768)
        hidden = model.vision.tower.vision_model(pixel_values=patches[None], spatial_shapes=torch.tensor([[ph, pw]]), pixel_attention_mask=torch.ones(1, ph * pw, dtype=torch.int32)).last_hidden_state
        outs.append(model.vision.projector(hidden.reshape(1, ph, pw, -1)))
    return torch.cat(outs, 1)


def audio_prefix(pcm):
    x = torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None]
    mel, frames = mel_frontend(x)
    h = model.audio.encoder(mel, int(frames[0]))
    return model.audio.residual(model.audio.adapter(h))


DELIM = {"state": "<|reserved_7|>", "q": "<|reserved_8|>", "opt": "<|reserved_9|>", "opt_end": "<|reserved_10|>", "decide": "<|reserved_11|>"}
MARKER = "<|mask|>"
QTYPES = {"choice": 0, "score": 1, "noul": 2}
esc = lambda t: re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", t)


def crit(v): return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q, noul_default, audio):
    if q["type"] == "choice":
        if audio: return [f"option_{i:03d}: {crit(k if v in (None, '') else v)}" for i, (k, v) in enumerate(q["criteria"].items())]
        return [k if v in (None, "") else f"{k}: {crit(v)}" for k, v in q["criteria"].items()]
    if q["type"] == "score": return [f"level {i}: {crit(c)}" for i, c in enumerate(q["criteria"])]
    if audio: return ["false: no", "true: yes"]
    c = q.get("criteria") or noul_default or {}
    f, t = c.get("false", c.get("no")), c.get("true", c.get("yes"))
    return ["false: " + (crit(f) if f not in (None, "") else "no, the statement does not hold"), "true: " + (crit(t) if t not in (None, "") else "yes, the statement holds")]


def encode(state, q, max_len, noul_default=None, audio=False, per_option=24):
    ids_of = tok.token_to_id
    enc = lambda s: tok.encode(esc(s), add_special_tokens=False).ids
    opts = render_options(q, noul_default, audio)
    budget = max(96, min(len(opts) * per_option + 32, max_len // 2)); per = max(2, (budget - 3 * len(opts)) // len(opts))
    question = ([ids_of(DELIM["q"])] + enc(q["instructions"]))[: max(16, budget)]
    markers = []
    for text in opts:
        markers.append(len(question) + 1)
        question += [ids_of(DELIM["opt"]), ids_of(MARKER)] + enc(" " + text)[:per] + [ids_of(DELIM["opt_end"])]
    question.append(ids_of(DELIM["decide"]))
    room = max(0, max_len - len(question) - 2)
    state_ids = [ids_of(DELIM["state"])] + enc(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))[:room]
    ids = ([cfg["bos_token_id"]] + state_ids + question)[:max_len]
    return ids, [m + 1 + len(state_ids) for m in markers]


def temperature_key(q):
    k = 2 if q["type"] == "noul" else len(q["criteria"])
    return f"{q['type']}:" + ("2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+")


def answer(state, questions, prefix, mode):
    max_len = {"text": cfg["max_length"], "image": cfg["image_text_length"], "audio": cfg["audio_text_length"]}[mode]
    p = 0 if prefix is None else prefix.shape[1]
    max_len = min(max_len, cfg["max_length"] - p)
    noul = None if mode == "text" else {"false": "no", "true": "yes"}
    if mode == "audio" and state is None: state = {}
    state = "" if state is None else state
    out = []
    for q in questions:
        ids, markers = encode(state, q, max_len, noul, mode == "audio")
        text = model.encoder.embed_tokens(torch.tensor(ids))
        seq = text if prefix is None else torch.cat([prefix[0], text])
        h = model.encoder(seq[None], torch.ones(1, len(seq), dtype=torch.bool), torch.tensor([p]))
        htext = h[:, p:]
        z = model.head(htext, torch.ones(1, len(ids), dtype=torch.bool), [torch.tensor(markers)], torch.tensor([QTYPES[q["type"]]]))[0]
        k = 2 if q["type"] == "noul" else len(q["criteria"])
        z = z[:k]
        if mode == "text": z = z / cfg["temperatures"].get(temperature_key(q), cfg["temperatures"].get(q["type"], 1.0))
        out.append(torch.softmax(z, -1).tolist())
    return out


h, w = 160, 224
yy, xx = np.mgrid[0:h, 0:w]
rgb = np.stack([xx * 255 // (w - 1), yy * 255 // (h - 1), ((xx + yy) * 255 // (w + h - 2))], -1).astype(np.uint8)
img = Image.fromarray(rgb)
buf = io.BytesIO(); img.save(buf, format="PNG"); png_b64 = base64.b64encode(buf.getvalue()).decode()
sr = 16000
tt = np.arange(2 * sr) / sr
wavf = (0.4 * np.sin(2 * np.pi * 440 * tt) * (tt < 1.0) + 0.4 * np.sin(2 * np.pi * 220 * tt) * (tt >= 1.0)).astype(np.float32)
pcm = (wavf * 32767).astype(np.int16)
wbuf = io.BytesIO()
with wave.open(wbuf, "wb") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(pcm.tobytes())
wav_b64 = base64.b64encode(wbuf.getvalue()).decode()

text_q = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
}
requests = [
    {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "questions": text_q},
    {"state": {"subject": "Login broken", "body": "I cannot sign in since the update; the app crashes on launch."},
     "questions": {"team": {"type": "choice", "instructions": "Route the ticket", "criteria": {"billing": None, "tech": None, "sales": None}},
                   "refund": {"type": "noul", "instructions": "Refund requested?", "criteria": {"true": "money back", "false": "no refund"}}}},
    {"state": "A photo.", "images": [png_b64],
     "questions": {"corner": {"type": "choice", "instructions": "What is the dominant colour at the top-left corner?", "criteria": {"red": None, "green": None, "blue": None, "black": None}},
                   "gradient": {"type": "noul", "instructions": "Is this a smooth gradient?"}}},
    {"state": None, "audio": wav_b64,
     "questions": {"speech": {"type": "noul", "instructions": "Does the clip contain speech?"},
                   "kind": {"type": "choice", "instructions": "What is in the clip?", "criteria": {"speech": "a person talking", "music": "music or a tone", "silence": "nothing"}}}},
]
rows = []
with torch.no_grad():
    for r in requests:
        if r.get("images"): prefix, mode = image_prefix(img), "image"
        elif r.get("audio"): prefix, mode = audio_prefix(pcm), "audio"
        else: prefix, mode = None, "text"
        probs = answer(r["state"], list(r["questions"].values()), prefix, mode)
        answers = {name: {"probs": p, "prefix_tokens": 0 if prefix is None else prefix.shape[1]} for (name, q), p in zip(r["questions"].items(), probs)}
        rows.append({"request": r, "answers": answers})
json.dump({"repo": repo, "rows": rows}, open(out, "w"), indent=1)
print("wrote", out)
for r in rows:
    for n, a in r["answers"].items():
        print(n, [round(p, 4) for p in a["probs"]], "prefix", a["prefix_tokens"])
