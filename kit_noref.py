# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:35:45 2025

@author: kaya-
"""

# DNA-Transformer PRNG (NO-REF / NO-TRAIN) — FINAL
# Decoder-only + RoPE + deterministic sampling + KV-cache (sliding window) + dynamic RoPE (pos_offset)
# -----------------------------------------------------------------------------
# FINAL FIXES
# - Deterministic SDPA backend lock (global)
# - RoPE absolute position continuity across KV prune via pos_offset
# - Memory-efficient DNA/bits accumulation (bytearray/array)
# - Default: USE_SHAKE=True for keystream/key extraction (domain-separated)
# - Model fingerprint (SHA-256 of state_dict) mixed into extractor
# -----------------------------------------------------------------------------

import os

# Optional: deterministic CUDA GEMM (must be set before torch import)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Recommended to set BEFORE running Python:
#   export PYTHONHASHSEED=0
#   export CUDA_LAUNCH_BLOCKING=1 (debug only)

import math, random, time, json, struct, hashlib, re, zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from array import array

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --------------------------- Determinism --------------------------------------
DETERMINISTIC = True
FORCE_CPU = False
DISABLE_TF32 = True

if DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if DISABLE_TF32:
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

# Deterministic SDPA lock (PyTorch 2.x)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

# --------------------------- Config -------------------------------------------
BALANCE_GLOBAL      = True
STRICT_QUOTA        = False
SOFT_BALANCE        = True
BALANCE_PERIOD      = 64
BALANCE_JITTER      = True
LAG1_DAMP           = 0.025
ALPHA_BASE          = 0.10
ALPHA_END           = 0.60
HOMOPOLYMER_MAX     = 5

# IMPORTANT (Final): keystream/key should be extracted (not raw bits)
USE_SHAKE           = False #bin ler önemli değilse yani kriptografik anahtar üretmiyorsan false veya true fark etmez...

KS_BYTES            = 64 * 1024 #Eğer key bini nıste sokacaksan 125_000 olmalı..
KEY_BYTES           = 32
RULE_TEMP_ADAPTIVE  = 4.0

# KV-cache sliding window size (should match cfg.block_size)
KV_CACHE_MAX_SIZE   = 128

# Optional: perf / stats
try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    psutil = None
    _HAVE_PSUTIL = False

try:
    from scipy.stats import binomtest, chisquare
    _HAVE_SCIPY = True
except Exception:
    from math import erfc
    _HAVE_SCIPY = False
    def binomtest(k, n, p=0.5):
        mu = n * p
        var = n * p * (1-p) + 1e-12
        z = abs(k - mu) / math.sqrt(var)
        class _Res:
            def __init__(self, pv): self.pvalue = pv
        return _Res(erfc(z / math.sqrt(2.0)))
    def chisquare(obs, exp):
        k = len(obs) - 1
        exp = [float(e) for e in exp]
        chi2 = sum((o - e)**2 / (e + 1e-12) for o, e in zip(obs, exp))
        if chi2 <= 0:
            p = 1.0
        else:
            w = (chi2 / k) ** (1.0/3.0)
            mu_w = 1.0 - 2.0/(9.0*k)
            sigma_w = math.sqrt(2.0/(9.0*k))
            z_w = (w - mu_w) / (sigma_w + 1e-12)
            p = 1.0 - 0.5*erfc(-z_w/math.sqrt(2.0))
        class _Res:
            def __init__(self, pv): self.pvalue = pv
        return _Res(p)

# --------------------------- Constants / helpers ------------------------------
DNA = ["A","C","G","T"]
VOCAB = {b:i for i,b in enumerate(DNA)}
INV_VOCAB = {i:b for b,i in VOCAB.items()}

MODEL_SEED = 1337  # deterministic model init

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(MODEL_SEED)

ENC_RULES: List[Dict[str,str]] = [
    {"A":"00","C":"01","G":"10","T":"11"},
    {"A":"11","C":"01","G":"10","T":"00"},
    {"A":"00","C":"10","G":"01","T":"11"},
    {"A":"11","C":"10","G":"01","T":"00"},
    {"A":"01","C":"00","G":"11","T":"10"},
    {"A":"10","C":"00","G":"11","T":"01"},
    {"A":"01","C":"11","G":"00","T":"10"},
    {"A":"10","C":"11","G":"00","T":"01"},
]
ENC_RULE_NAMES: List[str] = []
for i,rule in enumerate(ENC_RULES):
    parts = [f"{b}={bits}" for b,bits in rule.items()]
    ENC_RULE_NAMES.append(f"R{i}: " + " ".join(parts))

RULE_MATS = np.zeros((8,4,2), dtype=np.float32)
for r,rule in enumerate(ENC_RULES):
    for bch, bits in rule.items():
        b = VOCAB[bch]
        RULE_MATS[r,b,0] = 1.0 if bits[0] == '1' else 0.0
        RULE_MATS[r,b,1] = 1.0 if bits[1] == '1' else 0.0
RULE_MATS_T = torch.tensor(RULE_MATS)  # [8,4,2] on CPU by default

def make_generator(master_seed: int, device: str = 'cpu') -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(master_seed))
    return g

def sample_categorical(probs_1d: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    p = probs_1d / probs_1d.sum()
    cdf = torch.cumsum(p, dim=0)
    u = torch.rand((1,), generator=gen, device=p.device, dtype=p.dtype)
    idx = torch.searchsorted(cdf, u, right=False).clamp(max=p.numel()-1)
    return idx

@dataclass
class GPTConfig:
    vocab_size: int = 4
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0

        self.key   = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj  = nn.Linear(cfg.n_embd, cfg.n_embd)

        base = 10000.0
        d = self.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _apply_rope(self, x, start_index: int = 0):
        B,H,T,D = x.shape
        x = x.view(B,H,T,D//2,2)
        x1, x2 = x[...,0], x[...,1]
        device = x1.device
        dtype  = x1.dtype

        pos = torch.arange(start_index, start_index + T, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('t,f->tf', pos, self.inv_freq)
        cos = freqs.cos().to(dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype)[None, None, :, :]

        xr0 = x1 * cos - x2 * sin
        xr1 = x1 * sin + x2 * cos
        return torch.stack([xr0, xr1], dim=-1).reshape(B,H,T,D)

    def forward_with_past(self, x, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, pos_offset: int = 0):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim

        k = self.key(x).view(B, T, H, D).transpose(1, 2)
        q = self.query(x).view(B, T, H, D).transpose(1, 2)
        v = self.value(x).view(B, T, H, D).transpose(1, 2)

        past_len = 0 if (past_kv is None) else past_kv[0].size(2)
        start = pos_offset + past_len  # absolute position continuity

        q = self._apply_rope(q, start_index=start)
        k = self._apply_rope(k, start_index=start)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )
        y = attn.transpose(1,2).contiguous().view(B, T, H*D)
        return self.proj(y), (k, v)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4*cfg.n_embd),
            nn.GELU(),
            nn.Linear(4*cfg.n_embd, cfg.n_embd),
        )

    def forward_with_past(self, x, past_kv=None, pos_offset: int = 0):
        y, present = self.attn.forward_with_past(self.ln1(x), past_kv=past_kv, pos_offset=pos_offset)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, present

class MiniGPTDualHead(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.base_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.rule_head = nn.Linear(cfg.n_embd, 8, bias=False)

    def forward_with_past(self, idx, past_kv=None, pos_offset: int = 0):
        x = self.tok_emb(idx)
        presents = []
        for i, blk in enumerate(self.blocks):
            pkv_i = None if past_kv is None else past_kv[i]
            x, present_i = blk.forward_with_past(x, past_kv=pkv_i, pos_offset=pos_offset)
            presents.append(present_i)
        x = self.ln_f(x)
        return self.base_head(x), self.rule_head(x), presents

def prune_kv_cache(past_kv, max_size: int):
    pruned = []
    for K, V in past_kv:
        seq_len = K.size(2)
        if seq_len > max_size:
            pruned.append((K[:, :, -max_size:, :].contiguous(),
                           V[:, :, -max_size:, :].contiguous()))
        else:
            pruned.append((K, V))
    return pruned

def select_rule_adaptive_improved(
    p_base: torch.Tensor,
    rule_temp: float,
    recent_rules: List[int],
    gen: torch.Generator,
    RM: torch.Tensor
) -> int:
    # RM: [8,4,2] on correct device
    Eb = torch.einsum('f,rfj->rj', p_base, RM)  # [8,2] P(bit=1)

    eps = 1e-6
    p = Eb.clamp(eps, 1.0 - eps)
    ent = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))  # [8,2]
    entropy_scores = ent.sum(dim=1)  # [8]

    diversity = torch.ones(8, device=p_base.device)
    if recent_rules:
        recent = recent_rules[-8:]
        counts = torch.zeros(8, device=p_base.device)
        for r in recent:
            if 0 <= r < 8:
                counts[r] += 1
        diversity = 1.0 / (1.0 + counts)
        diversity = diversity / diversity.mean()

    score = entropy_scores * diversity
    pr = torch.softmax(score / max(1e-4, rule_temp), dim=-1)
    return int(sample_categorical(pr, gen).item())

def _compute_target_probs(gc_target: Optional[float]) -> Dict[str,float]:
    assert 0.0 <= gc_target <= 1.0
    pC = pG = gc_target/2.0
    pA = pT = (1.0-gc_target)/2.0
    probs = {"A":pA,"C":pC,"G":pG,"T":pT}
    s = sum(probs.values())
    return {k: float(v/s) for k,v in probs.items()}

@torch.no_grad()
def generate_constrained_with_rules(
    model: MiniGPTDualHead,
    start_tokens: torch.LongTensor,
    out_len: int,
    temperature: float,
    top_k: Optional[int],
    rule_temp: float,
    device: str,
    gc_target: float,
    homopolymer_max: Optional[int],
    balance_mode: str,
    rule_mode: str,
    rule_id: int,
    gen: torch.Generator
) -> Tuple[str, array, bytes]:
    model.eval()
    cfg = model.cfg

    # Cache RULE_MATS on device once (avoid per-step .to(device))
    RM = RULE_MATS_T.to(device)

    # Keep only prompt on device (no CPU bounce)
    idx = start_tokens.to(device)

    # memory-efficient accumulators
    dna_ascii = bytearray()
    bits_ascii = bytearray()
    rules = array('B')  # 0..7

    target = _compute_target_probs(gc_target) if balance_mode != 'none' else None

    if balance_mode == 'global':
        quotas = {b:int(round(out_len*target[b])) for b in DNA}
        diff = out_len - sum(quotas.values())
        if diff != 0:
            order_idx = torch.randperm(4, generator=gen, device=torch.device(device)).tolist()
            order = [DNA[i] for i in order_idx]
            for b in order:
                if diff == 0: break
                quotas[b] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
        remain = quotas.copy()
    else:
        remain = {b:10**18 for b in DNA}

    last_base = None
    run_len = 0
    pos_offset = 0

    base_logits, rule_logits, past = model.forward_with_past(idx, past_kv=None, pos_offset=pos_offset)

    for step in tqdm(range(out_len), desc="generate"):
        base_step = base_logits[:, -1, :].clone() / max(1e-4, temperature)
        p_base = torch.softmax(base_step, dim=-1)[0]

        # periodic weak GC mix
        if balance_mode != 'none' and target is not None:
            if BALANCE_JITTER:
                apply_mix = (torch.randint(0, BALANCE_PERIOD, (1,), generator=gen, device=p_base.device).item() == 0)
            else:
                apply_mix = ((step % BALANCE_PERIOD) == 0)
            if apply_mix:
                prog = step / max(1, out_len - 1)
                lam  = ALPHA_BASE + (ALPHA_END - ALPHA_BASE) * prog
                target_vec = torch.tensor([target['A'], target['C'], target['G'], target['T']],
                                          dtype=p_base.dtype, device=p_base.device)
                p_base = (1.0 - lam) * p_base + lam * target_vec
                p_base = p_base / p_base.sum()

        mask = torch.ones(4, dtype=torch.bool, device=p_base.device)

        # homopolymer constraint
        if homopolymer_max is not None and last_base is not None and run_len >= homopolymer_max:
            mask[last_base] = False

        # global quota / soft balance
        if balance_mode == 'global':
            if STRICT_QUOTA:
                quota_mask = torch.tensor([remain[INV_VOCAB[i]] > 0 for i in range(4)],
                                          device=p_base.device, dtype=torch.bool)
                mask = mask & quota_mask
            if SOFT_BALANCE:
                frac_done = step / max(1, out_len - 1)
                if BALANCE_JITTER:
                    apply_now = (torch.randint(0, BALANCE_PERIOD, (1,), generator=gen, device=p_base.device).item() == 0) or (frac_done > 0.95)
                else:
                    apply_now = ((step % BALANCE_PERIOD) == 0) or (frac_done > 0.95)
                if apply_now:
                    if frac_done < 0.90:
                        alpha = ALPHA_BASE
                    else:
                        alpha = ALPHA_BASE + (ALPHA_END - ALPHA_BASE) * (frac_done - 0.90) / 0.10
                    alpha = float(max(0.0, min(2.0, alpha)))
                    rem_pos = {b: max(0, remain[b]) for b in DNA}
                    rem_total = sum(rem_pos.values())
                    if rem_total > 0:
                        target_vec = torch.tensor([target[INV_VOCAB[i]] for i in range(4)],
                                                  device=p_base.device, dtype=p_base.dtype)
                        remain_vec = torch.tensor([rem_pos[INV_VOCAB[i]]/rem_total for i in range(4)],
                                                  device=p_base.device, dtype=p_base.dtype)
                        soft = torch.clamp(remain_vec / torch.clamp(target_vec, min=1e-9), min=1e-6)
                        soft = torch.pow(soft, alpha)
                        p_base = p_base * soft

        # lag-1 damp
        if last_base is not None and LAG1_DAMP > 0.0:
            p_base[last_base] = p_base[last_base] * (1.0 - LAG1_DAMP)
            s = p_base.sum()
            if s > 0:
                p_base = p_base / s

        # top-k
        if top_k is not None:
            _, idxk = torch.topk(p_base, k=min(top_k,4))
            keep = torch.zeros_like(p_base, dtype=torch.bool); keep[idxk] = True
            mask = mask & keep

        # apply mask + renorm
        p_base = p_base * mask.float()
        if p_base.sum() <= 0:
            p_base = torch.softmax(base_step[0], dim=-1) * mask.float()
            if p_base.sum() <= 0:
                mask = torch.ones_like(mask, dtype=torch.bool)
                p_base = torch.softmax(base_step[0], dim=-1)
        p_base = p_base / p_base.sum()

        # deterministic base sampling
        b = int(sample_categorical(p_base, gen).item())

        # rule selection
        if rule_mode == 'learned':
            rule_step = rule_logits[:, -1, :].clone() / max(1e-4, rule_temp)
            p_rule = torch.softmax(rule_step[0], dim=-1)
            r = int(sample_categorical(p_rule, gen).item())
        elif rule_mode == 'uniform':
            r = int(torch.randint(0, 8, (1,), generator=gen, device=p_base.device).item())
        elif rule_mode == 'adaptive':
            r = select_rule_adaptive_improved(p_base, rule_temp, list(rules)[-16:] if len(rules) > 0 else [], gen, RM)
        else:
            r = int(max(0, min(7, rule_id)))

        base_char = INV_VOCAB[b]
        bits2 = ENC_RULES[r][base_char]  # '00'/'01'/'10'/'11'

        # append outputs (memory-efficient)
        dna_ascii.append(ord(base_char))
        bits_ascii.extend(bits2.encode("ascii"))
        rules.append(r)

        # quota update
        if balance_mode == 'global':
            remain[base_char] -= 1

        # homopolymer run update
        if last_base is None or b != last_base:
            last_base = b; run_len = 1
        else:
            run_len += 1

        # one-token forward with KV-cache
        tok = torch.tensor([[b]], dtype=torch.long, device=device)
        base_logits, rule_logits, past = model.forward_with_past(tok, past_kv=past, pos_offset=pos_offset)

        # KV prune + pos_offset update
        if past is not None:
            seq_len = max(k.size(2) for k, v in past)
            if seq_len > KV_CACHE_MAX_SIZE:
                drop = seq_len - KV_CACHE_MAX_SIZE
                pos_offset += drop
                past = prune_kv_cache(past, max_size=KV_CACHE_MAX_SIZE)

    dna_seq = dna_ascii.decode("ascii")
    return dna_seq, rules, bytes(bits_ascii)

# --------------------------- Analysis helpers ---------------------------------
def _entropy_bits_per_base(counts: Dict[str,int]) -> float:
    N = sum(counts.values())
    if N == 0: return 0.0
    ps = [counts.get(b,0)/N for b in 'ACGT']
    return float(-sum(p*math.log(p,2) for p in ps if p>0))

def analyze_outputs(dna_seq: str, bits_ascii: bytes) -> Dict[str,object]:
    N = len(dna_seq)
    counts = {b: dna_seq.count(b) for b in 'ACGT'}
    ent = _entropy_bits_per_base(counts)

    ones = bits_ascii.count(ord('1'))
    n_bits = len(bits_ascii)
    p_mono = float(binomtest(ones, n_bits, p=0.5).pvalue) if N>0 else 1.0

    exp = [N/4.0]*4
    p_chi = float(chisquare([counts['A'],counts['C'],counts['G'],counts['T']], exp).pvalue) if N>0 else 1.0

    return {
        "len_bases": N,
        "bits_len": n_bits,
        "entropy_bits_per_base": ent,
        "monobit_p": p_mono,
        "chi2_p": p_chi,
        "counts": counts
    }

def pack_bits_ascii_to_bytes(bits_ascii: bytes) -> bytes:
    # bits_ascii: b'0101...'
    out = bytearray()
    acc = 0
    n = 0
    for ch in bits_ascii:
        acc = (acc << 1) | (1 if ch == ord('1') else 0)
        n += 1
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)

def model_fingerprint_sha256(model: nn.Module) -> bytes:
    h = hashlib.sha256()
    sd = model.state_dict()
    for name, t in sd.items():
        h.update(name.encode("utf-8") + b"\x00")
        h.update(str(tuple(t.shape)).encode("ascii") + b"\x00")
        h.update(str(t.dtype).encode("ascii") + b"\x00")
        tc = t.detach().contiguous().cpu()
        h.update(tc.numpy().tobytes(order="C"))
    return h.digest()

def keystream_from_bits_extractor(
    bits_ascii: bytes,
    run_seed: int,
    n_bytes: int,
    domain: bytes,
    model_fingerprint: Optional[bytes] = None
) -> bytes:
    packed = pack_bits_ascii_to_bytes(bits_ascii)
    shake = hashlib.shake_256()
    shake.update(domain + b"\x00")
    shake.update(struct.pack(">Q", int(run_seed)))
    if model_fingerprint:
        shake.update(b"\x01" + model_fingerprint)
    shake.update(b"\x02" + struct.pack(">Q", len(bits_ascii)))  # bit-length (in bits)
    shake.update(b"\x03" + packed)
    return shake.digest(n_bytes)

# --------------------------- I/O ----------------------------------------------
def save_outputs(
    tag: str,
    dna: str,
    rules: array,
    bits_ascii: bytes,
    out_dir: str = "outputs",
    run_seed: Optional[int] = None,
    model_seed: Optional[int] = None,
    device: Optional[str] = None,
    model_fingerprint_hex: Optional[str] = None
):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"{tag}_{ts}")

    dna_path   = base + ".dna.txt"
    bits_path  = base + ".bits.txt"
    rules_path = base + ".rules.txt"
    meta_path  = base + ".json"

    with open(dna_path, "w", encoding="utf-8") as f:
        f.write(dna)

    # write bits as bytes directly (ASCII '0'/'1')
    with open(bits_path, "wb") as f:
        f.write(bits_ascii)

    # rules as space-separated
    with open(rules_path, "w", encoding="utf-8") as f:
        # streaming-ish generation
        f.write(" ".join(str(int(x)) for x in rules))

    from collections import Counter
    rc = Counter(list(rules))
    counts = {i: int(rc.get(i,0)) for i in range(8)}
    top_r = max(counts, key=counts.get) if len(rules)>0 else 0

    meta = {
        "dna_len": len(dna),
        "bits_len": len(bits_ascii),
        "rule_counts": {str(i): counts.get(i,0) for i in range(8)},
        "most_used_rule": int(top_r),
        "most_used_rule_name": ENC_RULE_NAMES[top_r],
        "run_seed": int(run_seed) if run_seed is not None else None,
        "model_seed": int(model_seed) if model_seed is not None else None,
        "deterministic": bool(DETERMINISTIC),
        "device": str(device) if device is not None else None,
        "use_shake_extractor": bool(USE_SHAKE),
        "model_fingerprint_sha256": model_fingerprint_hex,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SUMMARY] rule usage={counts}; most used: R{top_r} -> {ENC_RULE_NAMES[top_r]}")
    print(f"Saved:\n  DNA   = {dna_path}\n  BITS  = {bits_path}\n  RULES = {rules_path}\n  META  = {meta_path}")

# --------------------------- RUN ----------------------------------------------
OUT_LEN = 500_000

if __name__ == "__main__":
    cfg = GPTConfig(block_size=KV_CACHE_MAX_SIZE)
    assert cfg.block_size == KV_CACHE_MAX_SIZE, "KV_CACHE_MAX_SIZE must match cfg.block_size for this setup."

    if FORCE_CPU:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # RUN_SEED single source (ENV override)
    run_seed = int.from_bytes(os.urandom(8), "little") & ((1<<63)-1)
    seed_env = os.getenv("RUN_SEED")
    if seed_env and seed_env.strip():
        run_seed = int(seed_env, 0)
        print(f"[RUN] run_seed(override)={run_seed}")

    gen = make_generator(run_seed, device=device)

    print(f"[RUN] run_seed={run_seed}  model_seed={MODEL_SEED}  device={device}  deterministic={DETERMINISTIC}")

    # deterministic model init
    seed_all(MODEL_SEED)
    model = MiniGPTDualHead(cfg).to(device).eval()

    # model fingerprint (for extractor binding)
    fp = model_fingerprint_sha256(model)
    fp_hex = fp.hex()
    print(f"[MODEL] fingerprint_sha256={fp_hex[:16]}...")

    # start prompt directly on device
    start = torch.randint(0, 4, (1, cfg.block_size), generator=gen, device=torch.device(device), dtype=torch.long)

    # ---- PROFILE START ----
    if _HAVE_PSUTIL:
        proc = psutil.Process(os.getpid())
        cpu0_user, cpu0_sys = proc.cpu_times().user, proc.cpu_times().system
        rss0 = proc.memory_info().rss
    else:
        cpu0_fallback = time.process_time()
        cpu0_user = cpu0_sys = 0.0
        rss0 = None
    t0_ns = time.perf_counter_ns()

    dna, rules, bits_ascii = generate_constrained_with_rules(
        model=model,
        start_tokens=start,
        out_len=OUT_LEN,
        temperature=1.0,
        top_k=None,
        rule_temp=RULE_TEMP_ADAPTIVE,
        device=device,
        gc_target=0.5,
        homopolymer_max=HOMOPOLYMER_MAX,
        balance_mode=("global" if BALANCE_GLOBAL else "none"),
        rule_mode="adaptive",
        rule_id=0,
        gen=gen
    )

    t1_ns = time.perf_counter_ns()
    if _HAVE_PSUTIL:
        cpu1_user, cpu1_sys = proc.cpu_times().user, proc.cpu_times().system
        rss1 = proc.memory_info().rss
        cpu_s = (cpu1_user + cpu1_sys) - (cpu0_user + cpu0_sys)
        ram_mb = (rss1 - rss0) / (1024*1024)
    else:
        cpu_s = time.process_time() - cpu0_fallback
        ram_mb = None

    t_core_ms = (t1_ns - t0_ns) / 1e6
    eff = (len(dna) / t_core_ms) if t_core_ms > 0 else float("inf")

    print("\n--- CORE PERFORMANCE (no I/O / no analysis) ---")
    ram_str = f"{ram_mb:.2f} MB" if ram_mb is not None else "N/A"
    print(f"Süre(ms)={t_core_ms:.3f}  CPU(s)={cpu_s:.3f}  RAM_delta={ram_str}  Verimlilik={eff:.2f} char/ms")

    stats = analyze_outputs(dna, bits_ascii)
    print("\n--- ANALYSIS ---")
    print(f"bases={stats['len_bases']:,}  bits={stats['bits_len']:,}  entropy(bits/base)={stats['entropy_bits_per_base']:.6f}")
    print(f"mono_p={stats['monobit_p']:.6g}  chi_p={stats['chi2_p']:.6g}")
    c = stats["counts"]; tot = stats["len_bases"]
    print(f"A={c['A']} ({100*c['A']/tot:.2f}%)  C={c['C']} ({100*c['C']/tot:.2f}%)  "
          f"G={c['G']} ({100*c['G']/tot:.2f}%)  T={c['T']} ({100*c['T']/tot:.2f}%)")

    # Homopolymer max run
    if len(dna) > 0:
        max_run = max(len(m.group(0)) for m in re.finditer(r"(A+|C+|G+|T+)", dna))
    else:
        max_run = 0
    print("max homopolymer run =", max_run)

    # Compressibility of bitstring
    comp = zlib.compress(bits_ascii, level=9) if len(bits_ascii)>0 else b""
    print("compression ratio =", (len(comp)/len(bits_ascii)) if len(bits_ascii)>0 else 0.0)

    # Keystream/key derivation
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)

    if USE_SHAKE:
        keystream = keystream_from_bits_extractor(bits_ascii, run_seed, KS_BYTES, domain=b"DNA-PRNG/core-v1", model_fingerprint=fp)
        key_256   = keystream_from_bits_extractor(bits_ascii, run_seed, KEY_BYTES, domain=b"DNA-PRNG/key-v1",  model_fingerprint=fp)
    else:
        raw = pack_bits_ascii_to_bytes(bits_ascii)
        keystream = raw[:KS_BYTES]
        key_256   = raw[:KEY_BYTES]

    ks_path = os.path.join("outputs", f"keystream_{ts}.bin")
    with open(ks_path, "wb") as f:
        f.write(keystream)
    print(f"\n[KEYSTREAM] {len(keystream)} bytes yazıldı -> {ks_path}")
    print(f"[KEYSTREAM] ilk 32 bayt (hex): {keystream[:32].hex()}")

    key_path = os.path.join("outputs", f"key_{ts}.hex")
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(key_256.hex() + "\n")
    print(f"[KEY] {8*len(key_256)}-bit (hex): {key_256.hex()}")
    print(f"[KEY SAVED] {key_path}")

    # Final save bundle
    save_outputs(
        tag="noref_transformer_effective_FINAL",
        dna=dna,
        rules=rules,
        bits_ascii=bits_ascii,
        out_dir="outputs",
        run_seed=run_seed,
        model_seed=MODEL_SEED,
        device=device,
        model_fingerprint_hex=fp_hex
    )