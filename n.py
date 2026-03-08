#!/usr/bin/env python3
# n.py — Joint AR+SAT Trainer with Expansion Ratio Testing
# Enhanced inference: checkpoint name, tok/s, UK time

from __future__ import annotations
import argparse, json, math, pathlib, random, time, os, sys
from contextlib import nullcontext
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import torch

STATUS_FILE = "/workspace/status.json"

def write_status(step, seen_tok, loss, batch, block, tok_per_sec, phase):
    import json as _json
    try:
        with open(STATUS_FILE, 'w') as f:
            _json.dump({"step": step, "seen_tok": seen_tok, "loss": float(loss) if loss else None, "batch": batch, "block": block, "tok_per_sec": tok_per_sec, "phase": phase, "updated": time.time(), "target_tok": 35737600000}, f)
    except: pass

def show_status():
    import json as _json
    try:
        with open(STATUS_FILE) as f:
            s = _json.load(f)
        age = time.time() - s.get("updated", 0)
        target = s.get("target_tok") or 35737600000
        remaining = target - s.get("seen_tok", 0)
        eta_sec = remaining / max(s.get("tok_per_sec", 1), 1)
        eta_days = eta_sec / 86400
        print(f"Step: {s.get('step', '?'):,} | Tokens: {s.get('seen_tok', 0)/1e9:.2f}B / {target/1e9:.1f}B | Loss: {s.get('loss', 0):.4f}")
        print(f"Speed: {s.get('tok_per_sec', 0):.0f} tok/s | B={s.get('batch')} L={s.get('block')} | ETA: {eta_days:.1f} days | {age:.0f}s ago")
    except FileNotFoundError:
        print("No status file. Training not running?")
    except Exception as e:
        print(f"Error: {e}")

# SafeProgress - Claude-safe progress (discrete lines, not single growing line)
class SafeProgress:
    def __init__(self, total, initial=0, unit="tok", print_every=500):
        self.total, self.n, self.unit = total, initial, unit
        self.last_print, self.postfix = initial, {}
        self.start_time = __import__('time').time()
    def update(self, n=1):
        self.n += n
        if self.n - self.last_print >= 1000000:  # print every ~1M tokens
            self._print(); self.last_print = self.n
    def set_postfix(self, **kwargs): self.postfix = kwargs
    def _print(self):
        elapsed = __import__('time').time() - self.start_time
        rate = self.n / elapsed if elapsed > 0 else 0
        pct = 100 * self.n / self.total if self.total > 0 else 0
        pf = ' '.join(f"{k}={v}" for k,v in self.postfix.items())
        print(f"[{pct:.1f}%] {self.n:,}/{self.total:,} {self.unit} | {rate:.0f} tok/s | {pf}")
    def close(self): self._print(); print("Done.")

import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, logging as hf_log

# ───────────────────────── ANSI Colors ─────────────────────────
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    PROMPT = "\033[36m"
    GEN = "\033[0m"
    INFO = "\033[90m"
    WARN = "\033[93m"

# ───────────────────────── Globals ─────────────────────────
hf_log.set_verbosity_error()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "deepseek-ai/DeepSeek-V3.2")
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "<|pad|>"})

VOCAB, EOS = (
    max(tok.get_vocab().values()) + 1,
    tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
)

# ───────────────────────── PRESETS ─────────────────────────
PRESETS: Dict[str, Dict[str, int]] = {
    "femto_1x":  dict(d=16, layers=1, heads=1, rank=16),
    "femto_12x": dict(d=16, layers=1, heads=1, rank=192),
    "femto_24x": dict(d=16, layers=1, heads=1, rank=384),
    "pico_1x":   dict(d=32, layers=1, heads=2, rank=16),
    "pico_3x":   dict(d=32, layers=1, heads=2, rank=48),
    "pico_6x":   dict(d=32, layers=1, heads=2, rank=96),
    "pico_12x":  dict(d=32, layers=1, heads=2, rank=192),
    "pico_24x":  dict(d=32, layers=1, heads=2, rank=384),
    "pico_48x":  dict(d=32, layers=1, heads=2, rank=768),
    "nano_1x":   dict(d=64,  layers=2, heads=4, rank=16),
    "nano_3x":   dict(d=64,  layers=2, heads=4, rank=48),
    "nano_6x":   dict(d=64,  layers=2, heads=4, rank=96),
    "nano_12x":  dict(d=64,  layers=2, heads=4, rank=192),
    "nano_24x":  dict(d=64,  layers=2, heads=4, rank=384),
    "nano_48x":  dict(d=64,  layers=2, heads=4, rank=768),
    "nano_96x":  dict(d=64,  layers=2, heads=4, rank=1536),
    "micro_3x":  dict(d=128, layers=4, heads=8, rank=48),
    "micro_6x":  dict(d=128, layers=4, heads=8, rank=96),
    "micro_12x": dict(d=128, layers=4, heads=8, rank=192),
    "micro_24x": dict(d=128, layers=4, heads=8, rank=384),
    "small":     dict(d=512, layers=8,  heads=16, rank=64),
    "smallx2":   dict(d=512, layers=16, heads=16, rank=64),
    "base":      dict(d=768, layers=12, heads=24, rank=96),
    "base18":    dict(d=768, layers=18, heads=24, rank=96),
    "large":     dict(d=1024, layers=24, heads=16, rank=128),
}

DEFAULT_BLOCK = 1122
DEFAULT_BATCH = 1
SAT_BLOCK = 2
LR_CORE, LR_HEAD = 5e-5, 2e-4
EMIT_LAMBDA = 0.1
DEFAULT_SAVE_SEC = 24 * 3600
CKDIR = pathlib.Path("ckpts_expansion")

DEFAULT_PRETRAIN_SOURCES = "OpenTransformer/goddess-crawl,OpenTransformer/agillm-crawl-data,OpenTransformer/web-crawl-2026,OpenTransformer/web-crawl-clean-v2,OpenTransformer/scraped-web-data,OpenTransformer/turbo-crawl,OpenTransformer/sft-data-clean,OpenTransformer/web-crawl-v1"
DEFAULT_AFTER_SFT_SOURCES = "mlabonne/opc-sft-stage2-chat,HuggingFaceH4/ultrachat_200k"
DEFAULT_AFTER_SFT_BLOCK = 1122

# ───────────────────────── UK Time Helper ─────────────────────────
def get_uk_time() -> str:
    utc_now = datetime.now(timezone.utc)
    year = utc_now.year
    march_last = datetime(year, 3, 31, 1, 0, tzinfo=timezone.utc)
    while march_last.weekday() != 6:
        march_last = march_last.replace(day=march_last.day - 1)
    oct_last = datetime(year, 10, 31, 1, 0, tzinfo=timezone.utc)
    while oct_last.weekday() != 6:
        oct_last = oct_last.replace(day=oct_last.day - 1)
    if march_last <= utc_now < oct_last:
        uk_offset = 1
        tz_name = "BST"
    else:
        uk_offset = 0
        tz_name = "GMT"
    from datetime import timedelta
    uk_time = utc_now + timedelta(hours=uk_offset)
    return uk_time.strftime(f'%Y-%m-%d %H:%M:%S {tz_name}')

# ───────────────────────── Utilities ─────────────────────────
def rng_state():
    if DEV.type == "cuda":
        try:
            return torch.cuda.get_rng_state(DEV)
        except TypeError:
            return torch.cuda.get_rng_state()
    return torch.get_rng_state()

def _is_probably_ckpt(path: pathlib.Path) -> bool:
    try:
        return path.is_file() and path.suffix == ".pt" and not path.name.endswith(".pt.tmp") and path.stat().st_size > (1<<20)
    except Exception:
        return False

def _resolve_ckpt(path: pathlib.Path) -> pathlib.Path | None:
    try:
        if path.is_dir():
            cands = sorted([p for p in path.glob("*.pt") if _is_probably_ckpt(p)],
                           key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0] if cands else None
        if path.suffix == ".tmp":
            solid = path.with_suffix("")
            return solid if _is_probably_ckpt(solid) else _resolve_ckpt(path.parent)
        return path if _is_probably_ckpt(path) else _resolve_ckpt(path.parent)
    except Exception:
        return None

def _try_load(path: pathlib.Path, map_location="cpu"):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[ckpt-skip] {path} not usable: {e}")
        return None

def _prune_checkpoints(save_dir: pathlib.Path, phase_name: str, max_ckpts: int):
    if max_ckpts is None or max_ckpts <= 0:
        return
    try:
        for tmp in save_dir.glob("*.pt.tmp"):
            try:
                tmp.unlink()
                print(f"  [prune] cleaned stale tmp {tmp.name}")
            except Exception:
                pass
        pattern = f"{phase_name}_step*.pt"
        ckpts = sorted(
            [p for p in save_dir.glob(pattern) if _is_probably_ckpt(p)],
            key=lambda p: p.stat().st_mtime
        )
        excess = len(ckpts) - max_ckpts
        if excess > 0:
            for p in ckpts[:excess]:
                try:
                    p.unlink()
                    print(f"  [prune] deleted old {p.name}")
                except Exception:
                    pass
    except Exception as e:
        print(f"[ckpt-prune] error: {e}")

def print_expansion_info(cfg: dict, tie_weights: bool = False):
    d_k = cfg["d"] // cfg["heads"]
    rank = cfg["rank"]
    ratio = rank / d_k
    regime = "COMPRESSION" if ratio < 1 else ("IDENTITY" if ratio == 1 else "EXPANSION")
    tie_str = "YES" if tie_weights else "NO"
    print(f"\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
    print(f"\u2502 TUNEABLE ATTENTION CONFIG               \u2502")
    print(f"\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
    print(f"\u2502 d_model: {cfg['d']:4d}  heads: {cfg['heads']:2d}  d_k: {d_k:3d}     \u2502")
    print(f"\u2502 layers: {cfg['layers']:4d}  tie_weights: {tie_str:3s}          \u2502")
    print(f"\u2502 rank: {rank:4d}  ratio: {ratio:.1f}x  [{regime:11s}] \u2502")
    print(f"\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")

# ───────────────────────── AMP helper ─────────────────────────
try:
    from torch.amp import autocast as _ac, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _ac, GradScaler

def _auto_amp_dtype():
    if DEV.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported(): return torch.bfloat16
            return torch.float16
        except Exception: return torch.float16
    return torch.float32

def amp(enabled: bool):
    if not (enabled and DEV.type == "cuda"): return nullcontext()
    try: return _ac(device_type="cuda", dtype=_auto_amp_dtype())
    except TypeError: return _ac(dtype=_auto_amp_dtype())

# ───────────────────────── Chat & Data Stream ─────────────────────────
def _coerce_role(r: str) -> str:
    r = (r or "").lower()
    if r in {"user", "human", "customer"}: return "user"
    if r in {"assistant", "gpt", "bot"}: return "assistant"
    if r in {"system", "context"}: return "system"
    return r or "user"

def _render_chat_text_from_ex(ex: dict, messages_key: str, add_generation_prompt: bool) -> Optional[str]:
    msgs = ex.get(messages_key)
    if msgs is None:
        for alt in ("conversations", "dialog", "turns"):
            if isinstance(ex.get(alt), list):
                msgs = ex[alt]; break
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        try:
            norm = []
            for m in msgs:
                role = _coerce_role(m.get("role", "")); content = m.get("content", m.get("text", ""))
                if not isinstance(content, str): continue
                norm.append({"role": role, "content": content})
            if not norm: return None
            return tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception: return None
    for a, b in (("prompt", "response"), ("instruction", "output"), ("question", "answer")):
        if isinstance(ex.get(a), str) and isinstance(ex.get(b), str):
            return f"User: {ex[a]}\nAssistant: {ex[b]}"
    return None

def _open_stream_one(ds_name: str, seed: int, streaming: bool = True):
    dc = DownloadConfig(max_retries=5, use_etag=True, resume_download=True)
    if ":" in ds_name: base, config = ds_name.split(":", 1)
    else: base, config = ds_name, None
    if not streaming:
        print(f"[download] Downloading {ds_name} (non-streaming)...")
    if base == "json":
        data_files = {"train": config}
        ds = load_dataset("json", data_files=data_files, split="train", streaming=streaming, download_config=dc)
    else:
        ds = load_dataset(base, config, split="train", streaming=streaming, download_config=dc) if config else \
             load_dataset(base, split="train", streaming=streaming, download_config=dc)
    if streaming:
        return iter(ds.shuffle(buffer_size=1000, seed=seed))
    else:
        print(f"[download] Got {len(ds):,} examples. Shuffling...")
        ds = ds.shuffle(seed=seed)
        return iter(ds)


# ─────────────────────────────── HOT DATASET CONFIG ───────────────────────────────
_HOT_CFG_PATH = pathlib.Path("/workspace/hot_config.json")
_hot_cache = {"mtime": 0, "data": {}}

def get_hot_datasets(default):
    try:
        if _HOT_CFG_PATH.exists():
            mt = _HOT_CFG_PATH.stat().st_mtime
            if mt > _hot_cache["mtime"]:
                _hot_cache["data"] = json.loads(_HOT_CFG_PATH.read_text())
                _hot_cache["mtime"] = mt
            cfg = _hot_cache["data"]
            if "datasets" in cfg:
                ds = cfg["datasets"]
                if isinstance(ds, list): ds = ",".join(ds)
                print(f"[HOT] Using: {ds[:60]}...")
                return ds
    except Exception as e:
        print(f"[HOT] Error: {e}")
    return default

def token_stream(ds_names: str, target: int, seed: int = 42,
                 chat: bool = False, chat_messages_key: str = "messages",
                 sft_add_generation_prompt: bool = False, dataset_field_text: str = "text",
                 streaming: bool = True):
    ds_names = get_hot_datasets(ds_names)
    sources = [s.strip() for s in ds_names.split(",") if s.strip()]
    if not sources: return
    src_idx = 0; emitted = 0; it = None; attempts = 0; backoff_base = 2.0
    while emitted < target:
        try:
            if it is None: it = _open_stream_one(sources[src_idx], seed, streaming=streaming)
            ex = next(it)
            text = None
            if isinstance(ex, dict):
                if chat:
                    text = _render_chat_text_from_ex(ex, chat_messages_key, sft_add_generation_prompt)
                if text is None:
                    if dataset_field_text and isinstance(ex.get(dataset_field_text), str):
                        text = ex[dataset_field_text]
                    elif isinstance(ex.get("text"), str):
                        text = ex["text"]
            if not isinstance(text, str):
                attempts = 0; continue
            enc = tok.encode(text)
            if EOS is not None and (len(enc) == 0 or enc[-1] != EOS):
                enc = enc + [EOS]
            for t in enc:
                yield t
                emitted += 1
                if emitted >= target: return
            attempts = 0
        except StopIteration:
            it = None; src_idx = (src_idx + 1) % len(sources)
        except Exception as e:
            attempts += 1
            sleep_s = min(60.0, backoff_base ** min(attempts, 6))
            print(f"[stream-retry] {sources[src_idx]} error: {type(e).__name__}, sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s); it = None
            if attempts % 2 == 0 and len(sources) > 1:
                src_idx = (src_idx + 1) % len(sources)

# ───────────────────────── ALiBi ─────────────────────────
@torch._dynamo.disable
def _alibi_slopes(n_heads: int):
    def pow2slopes(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer(): vals = pow2slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        vals = pow2slopes(closest)
        extra = pow2slopes(2 * closest)
        vals += extra[0::2][: n_heads - closest]
    return torch.tensor(vals, device=DEV).view(1, n_heads, 1, 1)

@torch._dynamo.disable
def alibi_bias(n_heads: int, n_tokens: int):
    i = torch.arange(n_tokens, device=DEV).view(1, 1, n_tokens, 1)
    j = torch.arange(n_tokens, device=DEV).view(1, 1, 1, n_tokens)
    dist = (j - i).clamp_min(0)
    return -_alibi_slopes(n_heads) * dist

# ───────────────────────── Model components ─────────────────────────
class TuneableAttentionMHA(nn.Module):
    def __init__(self, d: int, h: int, r: int, use_relpos: bool = True):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.r = h, d // h, r
        self.use_relpos = use_relpos
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.U = nn.Parameter(torch.randn(self.dk, r))
        nn.init.orthogonal_(self.U)
        self.proj = nn.Linear(h * self.dk, d, bias=False)
        self.drop = nn.Dropout(0.1)

    def _proj_qk(self, x):
        B, N, _ = x.shape
        return (x.view(B, N, self.h, self.dk).transpose(1, 2) @ self.U)

    def _reshape_v(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dk).transpose(1, 2)

    def forward(self, x, mask=None, rel_bias_tokens=None, kv_cache=None, use_cache=False):
        q = self._proj_qk(self.q(x))
        k_new = self._proj_qk(self.k(x))
        v_new = self._reshape_v(self.v(x))
        if kv_cache is None:
            k, v = k_new, v_new
        else:
            k_cached, v_cached = kv_cache
            if use_cache:
                k = torch.cat([k_cached, k_new], dim=2)
                v = torch.cat([v_cached, v_new], dim=2)
            else:
                k, v = k_new, v_new
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        if self.use_relpos and rel_bias_tokens is not None:
            att = att + alibi_bias(self.h, rel_bias_tokens).to(att.dtype)[:, :, -q.size(2):, :]
        if mask is not None:
            att = att + mask.to(att.dtype)
        z = (att.softmax(-1) @ v).transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        out = self.drop(self.proj(z))
        return (out, (k, v)) if use_cache else out


class Block(nn.Module):
    def __init__(self, d: int, h: int, r: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = TuneableAttentionMHA(d, h, r)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, x, mask, kv=None, use_cache=False, total_seq_len=None):
        if use_cache:
            y, new_kv = self.mha(self.ln1(x), mask, rel_bias_tokens=total_seq_len, kv_cache=kv, use_cache=True)
            x = x + y + self.ff(self.ln2(x + y))
            return x, new_kv
        else:
            n = x.size(1)
            x = x + self.mha(self.ln1(x), mask, rel_bias_tokens=n)
            return x + self.ff(self.ln2(x))


class Encoder(nn.Module):
    def __init__(self, cfg, tie_weights: bool = False):
        super().__init__()
        d, l, h, r = cfg["d"], cfg["layers"], cfg["heads"], cfg["rank"]
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([Block(d, h, r) for _ in range(l)])
        self.ln = nn.LayerNorm(d)
        self.tie_weights = tie_weights

    def forward(self, ids, mask, kv_caches=None, use_cache=False, total_seq_len=None):
        x = self.emb(ids)
        if not use_cache:
            for blk in self.blocks:
                x = blk(x, mask)
            return self.ln(x)
        new_kvs = []
        for i, blk in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches else None
            x, kv_out = blk(x, mask, kv, use_cache=True, total_seq_len=total_seq_len)
            new_kvs.append(kv_out)
        return self.ln(x), new_kvs


class ARHead(nn.Module):
    def __init__(self, d, tie_weights: bool = False, embedding_weight: nn.Parameter = None):
        super().__init__()
        self.tie_weights = tie_weights
        if tie_weights and embedding_weight is not None:
            self.proj = nn.Linear(d, VOCAB, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d, VOCAB)

    def forward(self, h):
        return self.proj(h)


class SATHead(nn.Module):
    def __init__(self, d, mode="var"):
        super().__init__()
        self.proj = nn.Linear(d, VOCAB)
        self.gate = nn.Linear(d, 2) if mode == "var" else None
    def forward(self, h_last):
        return self.proj(h_last), (self.gate(h_last[:, 0]) if self.gate else None)

# ───────────────────────── Masks ─────────────────────────
def causal_mask(n):
    return torch.triu(torch.full((1, 1, n, n), float("-inf"), device=DEV), 1)

def sat_mask(n, block=SAT_BLOCK):
    idx = torch.arange(n, device=DEV)
    grp = idx.unsqueeze(0) // block
    allow = (grp.T == grp) | (grp.T > grp)
    return torch.where(allow, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)

def sat_mask_cached(new_len: int, cached_len: int, block=SAT_BLOCK):
    total_len = cached_len + new_len
    mask = torch.zeros((1, 1, new_len, total_len), device=DEV)
    return mask


# ───────────────────────── Checkpoint helpers ─────────────────────────
def save_ckpt(path: pathlib.Path, core, ar_h, sat_h, opt, scaler, meta):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state = {
        "core": _strip_compiled_prefix(core.state_dict()), "ar": _strip_compiled_prefix(ar_h.state_dict()), "sat": _strip_compiled_prefix(sat_h.state_dict()),
        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        "cfg": meta.get("cfg"), "tokenizer_id": TOKENIZER_ID,
        "tie_weights": meta.get("tie_weights", False),
        **{k: v for k, v in meta.items() if k not in ("cfg", "tie_weights")}
    }
    torch.save(state, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(path)
    (path.parent / "latest.json").write_text(json.dumps({"path": str(path), "step": meta["step"], "block_size": meta.get("block_size"), "batch_size": meta.get("batch_size"), "seen_tok": meta.get("seen_tok")}))
    print(f"\n\u2713 saved checkpoint {path.name}")

def _strip_compiled_prefix(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

def load_ckpt(path, core, ar_h, sat_h, opt, scaler):
    p = _resolve_ckpt(path) or path
    ck = _try_load(p, map_location="cpu")
    if ck is None: raise FileNotFoundError(f"No valid checkpoint at {p}")
    core.load_state_dict(_strip_compiled_prefix(ck["core"]))
    ar_h.load_state_dict(_strip_compiled_prefix(ck["ar"]))
    sat_h.load_state_dict(_strip_compiled_prefix(ck["sat"]))
    try: opt.load_state_dict(ck["opt"])
    except Exception: pass
    if ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
    return ck.get("step", 0), ck.get("seen_tok", 0), ck.get("wall_time", time.time()), ck.get("block_size")

def _safe_load_any(path: pathlib.Path, tgt: nn.Module, key: str | None = None):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return 0
    ck = _try_load(p, map_location="cpu")
    if ck is None: return 0
    sd = ck.get(key, ck) if key else ck
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    tgt_sd = tgt.state_dict()
    filt = {k: v for k, v in sd.items() if k in tgt_sd and v.shape == tgt_sd[k].shape}
    if filt: tgt.load_state_dict(filt, strict=False)
    return len(filt)

def infer_cfg_from_ckpt(path: pathlib.Path):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return None
    sd = _try_load(p, map_location="cpu")
    if sd is None: return None
    if "cfg" in sd: return dict(sd["cfg"])
    return None


# ───────────────────────── Training Logic ─────────────────────────
def _parse_grow_plan(s: str) -> List[int]:
    return sorted(set([int(x.strip()) for x in s.split(",") if x.strip() and int(x.strip()) >= 128]))

def _count_enabled_params(*modules) -> int:
    seen_data_ptrs = set()
    total = 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.data_ptr() not in seen_data_ptrs:
                seen_data_ptrs.add(p.data_ptr())
                total += p.numel()
    return total

def _phase_freeze(core: nn.Module, *, freeze_core: bool, unfreeze_ln: bool, train_emb: bool):
    for p in core.parameters(): p.requires_grad = not freeze_core
    if freeze_core:
        if unfreeze_ln:
            for blk in core.blocks:
                for p in blk.ln1.parameters(): p.requires_grad = True
                for p in blk.ln2.parameters(): p.requires_grad = True
            for p in core.ln.parameters(): p.requires_grad = True
        if train_emb:
            for p in core.emb.parameters(): p.requires_grad = True

def _train_phase(
    args, phase_name: str,
    core, ar_h, sat_h, opt, scaler,
    start_step, seen_tok, resume_wall_time,
    cfg, source, steps, block_size, batch_size,
    chat_cfg: dict,
    max_ckpts: int,
    target_tokens_override: Optional[int] = None,
    tie_weights: bool = False,
    streaming: bool = True
):
    BLOCK = block_size
    BATCH = batch_size
    if target_tokens_override is not None:
        target_tokens = target_tokens_override
    else:
        ratio = 51.2 if args.chilla_max_double else 25
        param_count = _count_enabled_params(core, ar_h, sat_h)
        target_tokens = int(ratio * param_count)
    if steps:
        phase_target_tokens = steps * BLOCK * BATCH
        total_tokens_needed = seen_tok + phase_target_tokens
    else:
        total_tokens_needed = target_tokens
        if total_tokens_needed <= seen_tok:
            print(f"[{phase_name}] target {total_tokens_needed} already reached.")
            return start_step, seen_tok, resume_wall_time
    stream = token_stream(
        source, total_tokens_needed, seed=42,
        chat=chat_cfg.get("chat", False),
        chat_messages_key=chat_cfg.get("key", "messages"),
        sft_add_generation_prompt=chat_cfg.get("gen_prompt", False),
        dataset_field_text=chat_cfg.get("text_field", "text"),
        streaming=streaming
    )
    ce_tok = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_gate = nn.CrossEntropyLoss()
    pbar = SafeProgress(total=total_tokens_needed, initial=seen_tok, unit="tok")
    grow_plan = _parse_grow_plan(args.grow_plan) if args.auto_grow else []
    buf: list[int] = []
    batch_accum: list[list[int]] = []
    step = start_step
    steps_since_last_grow = 0
    oom_retries = 0
    MAX_OOM_RETRIES = 2
    now_wall = time.time()
    last_save_mono = time.monotonic() - (now_wall - (resume_wall_time or now_wall))
    print(f"[{phase_name}] Starting. Goal: {total_tokens_needed:,} tokens. Batch={BATCH}, Block={BLOCK}")
    print(f"[{phase_name}] AR_ONLY={args.ar_only}, TIE_WEIGHTS={tie_weights}, STREAMING={streaming}")
    step_start_time = time.monotonic()
    tok_per_sec_avg = 0.0
    while seen_tok < total_tokens_needed:
        try:
            while len(buf) < BLOCK:
                buf.append(next(stream))
        except StopIteration:
            break
        seq = buf[:BLOCK]
        buf = buf[BLOCK:]
        batch_accum.append(seq)
        if len(batch_accum) < BATCH:
            continue
        ids = torch.tensor(batch_accum, device=DEV)
        batch_accum = []
        tgt_ar = ids.clone()
        try:
            with amp(args.amp):
                h_ar = core(ids, causal_mask(ids.size(1)))
                logits_ar = ar_h(h_ar)[:, :-1]
                loss_ar = ce_tok(logits_ar.reshape(-1, VOCAB), tgt_ar[:, 1:].reshape(-1))
                if args.ar_only:
                    loss = loss_ar
                else:
                    h_sat = core(ids, sat_mask(ids.size(1)))
                    logits_sat, gate = sat_h(h_sat[:, -SAT_BLOCK:])
                    tgt_sat = ids[:, 1:SAT_BLOCK+1]
                    loss_sat = ce_tok(logits_sat.reshape(-1, VOCAB), tgt_sat.reshape(-1))
                    if gate is not None:
                        loss_sat += EMIT_LAMBDA * ce_gate(gate, torch.ones(ids.size(0), device=DEV, dtype=torch.long))
                    loss = loss_ar + loss_sat
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                batch_accum = []
                opt.zero_grad(set_to_none=True)
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                oom_retries += 1
                if oom_retries <= MAX_OOM_RETRIES:
                    print(f"\n[{phase_name} OOM] Retry {oom_retries}/{MAX_OOM_RETRIES} at Batch={BATCH}, clearing VRAM...")
                    time.sleep(4)
                    continue
                oom_retries = 0
                if BATCH > 1:
                    print(f"\n[{phase_name} OOM] Reducing Batch: {BATCH} -> {BATCH - 1} (after {MAX_OOM_RETRIES} retries)")
                    BATCH -= 1
                    time.sleep(4)
                else:
                    if grow_plan:
                        smaller = [b for b in grow_plan if b < BLOCK]
                        new_block = smaller[-1] if smaller else max(128, BLOCK // 2)
                    else:
                        new_block = max(128, BLOCK // 2)
                    print(f"\n[{phase_name} OOM] Reducing Block: {BLOCK} -> {new_block}")
                    BLOCK = new_block
                    time.sleep(4)
                steps_since_last_grow = 0
                continue
            raise
        step += 1
        oom_retries = 0
        toks_processed = BLOCK * BATCH
        seen_tok += toks_processed
        pbar.update(toks_processed)
        pbar.set_postfix(loss=f"{loss.item():.3f}", B=BATCH, L=BLOCK)
        step_elapsed = time.monotonic() - step_start_time
        tok_per_sec_now = toks_processed / step_elapsed if step_elapsed > 0 else 0
        tok_per_sec_avg = 0.9 * tok_per_sec_avg + 0.1 * tok_per_sec_now if tok_per_sec_avg > 0 else tok_per_sec_now
        step_start_time = time.monotonic()
        write_status(step, seen_tok, loss.item(), BATCH, BLOCK, tok_per_sec_avg, phase_name)
        if args.save_every_sec > 0:
            now_mono = time.monotonic()
            if now_mono - last_save_mono >= args.save_every_sec:
                ck_name = f"{phase_name}_step{step:08d}.pt"
                save_ckpt(pathlib.Path(args.save_dir) / ck_name, core, ar_h, sat_h, opt, scaler,
                          meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights, "block_size": BLOCK, "batch_size": BATCH})
                _prune_checkpoints(pathlib.Path(args.save_dir), phase_name, max_ckpts)
                last_save_mono = now_mono
        if args.auto_grow:
            steps_since_last_grow += 1
            if steps_since_last_grow >= args.grow_every_steps:
                steps_since_last_grow = 0
                try:
                    idx = grow_plan.index(BLOCK)
                    if idx + 1 < len(grow_plan):
                        BLOCK = grow_plan[idx + 1]
                        print(f"[{phase_name} Grow] Block -> {BLOCK}")
                        if DEV.type == "cuda": torch.cuda.empty_cache()
                except ValueError:
                    grow_plan = sorted(set(grow_plan + [BLOCK]))
    pbar.close()
    save_ckpt(pathlib.Path(args.save_dir) / f"{phase_name}_final.pt", core, ar_h, sat_h, opt, scaler,
              meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights, "block_size": BLOCK, "batch_size": BATCH})
    return step, seen_tok, time.time()


# ───────────────────────── Main Orchestrator ─────────────────────────
def train(args):
    cfg = PRESETS[args.preset].copy()
    tie_weights = args.tie_weights
    print_expansion_info(cfg, tie_weights)
    if not args.fresh:
        src_probe = pathlib.Path(args.warmstart_from) if args.warmstart_from else pathlib.Path(args.save_dir) / "final.pt"
        prev_cfg = infer_cfg_from_ckpt(src_probe)
    else: prev_cfg = None
    if prev_cfg:
        cfg.update({k: v for k, v in prev_cfg.items() if k in cfg})
        if args.x2 and prev_cfg.get("layers"): cfg["layers"] = max(cfg["layers"], prev_cfg["layers"] * 2)
    if args.rank: cfg["rank"] = args.rank
    if args.x2 and not prev_cfg: cfg["layers"] *= 2
    print(f"Config: {cfg}")
    core = Encoder(cfg, tie_weights=tie_weights).to(DEV)
    ar_h = ARHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV)
    sat_h = SATHead(cfg["d"], mode="var").to(DEV)
    total_params = _count_enabled_params(core, ar_h, sat_h)
    print(f"Total parameters: {total_params:,}")
    if tie_weights:
        print(f"{Colors.WARN}[weight-tying] Embedding and LM head share weights{Colors.RESET}")
    if not args.fresh:
        src = pathlib.Path(args.warmstart_from) if args.warmstart_from else pathlib.Path(args.save_dir) / "final.pt"
        src = _resolve_ckpt(src)
        if src:
            loaded = _safe_load_any(src, core, key="core")
            _safe_load_any(src, ar_h, key="ar")
            _safe_load_any(src, sat_h, key="sat")
            if loaded: print(f"Warm-start loaded from {src}")
    _phase_freeze(core, freeze_core=args.freeze_core, unfreeze_ln=args.unfreeze_ln, train_emb=args.train_emb)
    opt = torch.optim.AdamW([
        {"params": [p for p in core.parameters() if p.requires_grad], "lr": args.lr_core},
        {"params": ar_h.parameters(), "lr": args.lr_head},
        {"params": sat_h.parameters(), "lr": args.lr_head},
    ])
    scaler = GradScaler(enabled=(args.amp and DEV.type == "cuda"))
    start_step, seen_tok, last_wall, _resumed_block = 0, 0, None, None
    if args.resume and not args.fresh:
        start_step, seen_tok, last_wall, _resumed_block = load_ckpt(pathlib.Path(args.resume), core, ar_h, sat_h, opt, scaler)
        print(f"Resumed from step {start_step}" + (f", block_size={_resumed_block}" if _resumed_block else ""))
    if args.compile:
        print("[torch.compile] Compiling model...")
        core = torch.compile(core, mode="reduce-overhead")
        ar_h = torch.compile(ar_h, mode="reduce-overhead")
        sat_h = torch.compile(sat_h, mode="reduce-overhead")
        print("[torch.compile] Done.")
    step, seen_tok, last_wall = _train_phase(
        args, "pretrain", core, ar_h, sat_h, opt, scaler,
        start_step, seen_tok, last_wall, cfg,
        args.source, args.steps,
        (_resumed_block if _resumed_block and args.auto_grow else None) or args.block or DEFAULT_BLOCK,
        args.batch_size or DEFAULT_BATCH,
        chat_cfg={"chat": args.chat, "key": args.chat_messages_key, "gen_prompt": args.sft_add_generation_prompt, "text_field": args.dataset_field_text},
        max_ckpts=args.max_ckpts,
        target_tokens_override=args.target_tokens,
        tie_weights=tie_weights
    )
    if (not args.after_sft_source) and (args.after_sft_steps and args.after_sft_steps > 0):
        args.after_sft_source = DEFAULT_AFTER_SFT_SOURCES
        args.after_sft_chat = True
        if args.after_sft_add_generation_prompt is None: args.after_sft_add_generation_prompt = True
        if not args.after_sft_block: args.after_sft_block = DEFAULT_AFTER_SFT_BLOCK
    if args.after_sft_source and args.after_sft_steps and args.after_sft_steps > 0:
        print("\n[Orchestrator] Starting Post-Pretraining SFT Phase...")
        _phase_freeze(core,
                       freeze_core=args.after_sft_freeze_core,
                       unfreeze_ln=args.after_sft_unfreeze_ln,
                       train_emb=args.after_sft_train_emb)
        opt = torch.optim.AdamW([
            {"params": [p for p in core.parameters() if p.requires_grad], "lr": args.after_sft_lr_core or args.lr_core},
            {"params": ar_h.parameters(), "lr": args.after_sft_lr_head or args.lr_head},
            {"params": sat_h.parameters(), "lr": args.after_sft_lr_head or args.lr_head},
        ])
        step, seen_tok, last_wall = _train_phase(
            args, "sft", core, ar_h, sat_h, opt, scaler,
            step, seen_tok, last_wall, cfg,
            args.after_sft_source, args.after_sft_steps,
            args.after_sft_block or DEFAULT_AFTER_SFT_BLOCK,
            args.batch_size or DEFAULT_BATCH,
            chat_cfg={
                "chat": args.after_sft_chat,
                "key": args.after_sft_chat_messages_key,
                "gen_prompt": args.after_sft_add_generation_prompt if args.after_sft_add_generation_prompt is not None else args.sft_add_generation_prompt,
                "text_field": args.after_sft_dataset_field_text
            },
            max_ckpts=args.max_ckpts,
            target_tokens_override=None,
            tie_weights=tie_weights,
            streaming=False
        )
    save_ckpt(pathlib.Path(args.save_dir) / "final.pt", core, ar_h, sat_h, opt, scaler,
              meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights, "block_size": args.block or DEFAULT_BLOCK, "batch_size": args.batch_size or DEFAULT_BATCH})
    print("\U0001f389 All Training Complete")


# ───────────────────────── Sampling ─────────────────────────
def _apply_penalties(logits, ids, n, rep_p, pres_p, freq_p):
    if ids.numel() == 0: return logits
    hist = ids[0, -n:].long() if n > 0 else ids[0].long()
    uniq, counts = torch.unique(hist, return_counts=True)
    if pres_p or freq_p:
        logits[..., uniq] -= (pres_p + freq_p * counts.to(logits.dtype))
    if rep_p != 1.0:
        sel = logits[..., uniq]
        logits[..., uniq] = torch.where(sel > 0, sel / rep_p, sel * rep_p)
    return logits

def _sample(logits, T, top_k, top_p, min_p, greedy):
    if greedy: return logits.argmax(-1, keepdim=True)
    probs = (logits / max(T, 1e-8)).softmax(-1)
    if top_k:
        v, i = torch.topk(probs, min(top_k, probs.size(-1)))
        probs = torch.zeros_like(probs).scatter_(-1, i, v)
    if top_p < 1.0:
        s_probs, s_idx = torch.sort(probs, descending=True, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, s_idx, s_probs * (torch.cumsum(s_probs, -1) <= top_p).to(probs.dtype))
    if min_p > 0: probs[probs < min_p] = 0
    if probs.sum() == 0: return logits.argmax(-1, keepdim=True)
    return probs.div_(probs.sum()).multinomial(1)

@torch.no_grad()
def infer(args):
    if args.mode == "ar":
        if args.temperature is None: args.temperature = 0.7
        if args.top_k is None: args.top_k = 0
        if args.repetition_penalty is None: args.repetition_penalty = 1.3
        if args.presence_penalty is None: args.presence_penalty = 0.0
        if args.frequency_penalty is None: args.frequency_penalty = 0.3
        if args.penalty_last_n is None: args.penalty_last_n = 128
        if args.var is None: args.var = False
    else:
        if args.temperature is None: args.temperature = 0.5
        if args.top_k is None: args.top_k = 30
        if args.repetition_penalty is None: args.repetition_penalty = 2.0
        if args.presence_penalty is None: args.presence_penalty = 0.6
        if args.frequency_penalty is None: args.frequency_penalty = 1.0
        if args.penalty_last_n is None: args.penalty_last_n = 200
        if args.var is None: args.var = True
    path = _resolve_ckpt(pathlib.Path(args.ckpt)) or pathlib.Path(args.ckpt)
    sd = torch.load(path, map_location="cpu")
    cfg = sd["cfg"]
    tie_weights = sd.get("tie_weights", False)
    uk_time = get_uk_time()
    ckpt_name = path.name
    print(f"\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
    print(f"\u2502 INFERENCE @ {uk_time:<35s} \u2502")
    print(f"\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
    print(f"\u2502 Checkpoint: {ckpt_name:<35s} \u2502")
    print(f"\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
    print_expansion_info(cfg, tie_weights)
    _use_fp16 = getattr(args, 'fp16', False)
    core = Encoder(cfg, tie_weights=tie_weights)
    ar_h = ARHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None)
    sat_h = SATHead(cfg["d"])
    core.load_state_dict(sd["core"])
    ar_h.load_state_dict(sd["ar"])
    sat_h.load_state_dict(sd["sat"])
    if _use_fp16:
        core.half(); ar_h.half(); sat_h.half()
        print(f"{Colors.INFO}Using fp16 inference{Colors.RESET}")
    core.to(DEV).eval()
    ar_h.to(DEV).eval()
    sat_h.to(DEV).eval()
    total_params = _count_enabled_params(core, ar_h, sat_h)
    if total_params >= 1_000_000_000:
        param_str = f"{total_params / 1_000_000_000:.2f}B"
    elif total_params >= 1_000_000:
        param_str = f"{total_params / 1_000_000:.2f}M"
    elif total_params >= 1_000:
        param_str = f"{total_params / 1_000:.2f}K"
    else:
        param_str = f"{total_params}"
    print(f"Model size: {param_str} parameters ({total_params:,})")
    prompt_tokens = tok.encode(args.prompt)
    prompt_len = len(prompt_tokens)
    ids = torch.tensor([prompt_tokens], device=DEV)
    if ids.size(1) == 0:
        ids = torch.tensor([[EOS]], device=DEV)
        prompt_len = 1
    mode_str = args.mode
    if args.mode == "sat":
        mode_str = f"sat-{'var' if args.var else 'fixed'}"
    print(f"{Colors.INFO}Generating ({mode_str})...{Colors.RESET}")
    start = time.time()
    if args.mode == "ar":
        h, kvs = core(ids, causal_mask(ids.size(1)), use_cache=True, total_seq_len=ids.size(1))
        for _ in range(args.max_new):
            logits = ar_h(h)[:, -1]
            logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
            nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
            ids = torch.cat([ids, nxt], 1)
            h, kvs = core(ids[:, -1:], None, kv_caches=kvs, use_cache=True, total_seq_len=ids.size(1))
    else:
        cached_len = ids.size(1)
        h, kvs = core(ids, sat_mask(ids.size(1)), use_cache=True, total_seq_len=cached_len)
        added = 0
        while added < args.max_new:
            logits_all, gate = sat_h(h[:, -SAT_BLOCK:])
            stride = SAT_BLOCK if (not args.var or gate is None) else (gate.softmax(-1).multinomial(1).item() + 1)
            new_tokens = []
            for i in range(int(stride)):
                logits = logits_all[:, i]
                logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
                nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
                new_tokens.append(nxt)
                ids = torch.cat([ids, nxt], 1)
                added += 1
                if added >= args.max_new: break
            if added >= args.max_new: break
            new_ids = torch.cat(new_tokens, dim=1)
            mask = sat_mask_cached(new_ids.size(1), cached_len)
            h, kvs = core(new_ids, mask, kv_caches=kvs, use_cache=True, total_seq_len=ids.size(1))
            cached_len = ids.size(1)
    elapsed = time.time() - start
    gen_tokens = len(ids[0]) - prompt_len
    tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
    all_tokens = ids[0].tolist()
    prompt_text = tok.decode(all_tokens[:prompt_len], skip_special_tokens=True)
    gen_text = tok.decode(all_tokens[prompt_len:], skip_special_tokens=True)
    print(f"{Colors.PROMPT}{prompt_text}{Colors.RESET}{gen_text}")
    print(f"{Colors.INFO}[{elapsed:.2f}s | {gen_tokens} tokens | {tok_per_sec:.1f} tok/s]{Colors.RESET}")


# ───────────────────────── CLI ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="AGILLM Expansion Ratio Testing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--preset", choices=PRESETS.keys(), default="nano_3x")
    tr.add_argument("--rank", type=int)
    tr.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    tr.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    tr.add_argument("--source", default=DEFAULT_PRETRAIN_SOURCES)
    tr.add_argument("--target_tokens", type=int)
    tr.add_argument("--steps", type=int)
    tr.add_argument("--amp", action="store_true")
    tr.add_argument("--compile", action="store_true", help="Use torch.compile for speedup")
    tr.add_argument("--save_every_sec", type=int, default=DEFAULT_SAVE_SEC)
    tr.add_argument("--save_dir", default=str(CKDIR))
    tr.add_argument("--resume", type=str)
    tr.add_argument("--x2", action="store_true")
    tr.add_argument("--warmstart_from", type=str)
    tr.add_argument("--fresh", action="store_true")
    tr.add_argument("--max_ckpts", type=int, default=None)
    tr.add_argument("--chilla_max_double", action="store_true")
    tr.add_argument("--tie_weights", action="store_true")
    tr.add_argument("--ar_only", action="store_true")
    tr.add_argument("--freeze_core", action="store_true")
    tr.add_argument("--unfreeze_ln", action="store_true")
    tr.add_argument("--train_emb", action="store_true")
    tr.add_argument("--lr_core", type=float, default=LR_CORE)
    tr.add_argument("--lr_head", type=float, default=LR_HEAD)
    tr.add_argument("--chat", action="store_true")
    tr.add_argument("--chat_messages_key", default="messages")
    tr.add_argument("--dataset_field_text", default="text")
    tr.add_argument("--sft_add_generation_prompt", action="store_true")
    tr.add_argument("--auto_grow", action="store_true")
    tr.add_argument("--grow_plan", default="576,640,768,896,1024,1122")
    tr.add_argument("--grow_every_steps", type=int, default=50000)
    tr.add_argument("--after_sft_source", default="")
    tr.add_argument("--after_sft_steps", type=int, default=0)
    tr.add_argument("--after_sft_chat", action="store_true")
    tr.add_argument("--after_sft_chat_messages_key", default="messages")
    tr.add_argument("--after_sft_dataset_field_text", default="text")
    tr.add_argument("--after_sft_add_generation_prompt", type=bool, default=None)
    tr.add_argument("--after_sft_block", type=int, default=0)
    tr.add_argument("--after_sft_freeze_core", action="store_true")
    tr.add_argument("--after_sft_unfreeze_ln", action="store_true")
    tr.add_argument("--after_sft_train_emb", action="store_true")
    tr.add_argument("--after_sft_lr_core", type=float, default=0.0)
    tr.add_argument("--after_sft_lr_head", type=float, default=0.0)
    inf = sub.add_parser("infer")
    inf.add_argument("--mode", choices=["ar", "sat"], required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--prompt", required=True)
    inf.add_argument("--max_new", type=int, default=120)
    inf.add_argument("--temperature", type=float, default=None)
    inf.add_argument("--greedy", action="store_true")
    inf.add_argument("--top_k", type=int, default=None)
    inf.add_argument("--top_p", type=float, default=0.9)
    inf.add_argument("--min_p", type=float, default=0.0)
    inf.add_argument("--repetition_penalty", type=float, default=None)
    inf.add_argument("--presence_penalty", type=float, default=None)
    inf.add_argument("--frequency_penalty", type=float, default=None)
    inf.add_argument("--penalty_last_n", type=int, default=None)
    inf.add_argument("--var", action="store_true", default=None)
    inf.add_argument("--no-var", dest="var", action="store_false")
    inf.add_argument("--fp16", action="store_true", help="Use fp16 inference (faster, lower VRAM)")
    st = sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "train": train(args)
    elif args.cmd == "status": show_status()
    else: infer(args)


if __name__ == "__main__":
    main()
