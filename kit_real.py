# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 14:34:52 2025

@author: kaya-
"""

# kit_real.py
# DNA-Transformer PRNG — TRAIN ONCE, GENERATE MANY
# ------------------------------------------------
# Subcommands:
#   TRAIN:      python kit_real.py train --real real_data_100k.txt --ckpt checkpoints/dna_prng.pt
#   GEN:        python kit_real.py gen --ckpt checkpoints/dna_prng.pt --real real_data_100k.txt --bits 1000000 --seed 12345 --out outputs/bits_S12345.txt
#   GEN-MANY:   python kit_real.py gen-many --ckpt checkpoints/dna_prng.pt --real real_data_100k.txt --bits 1000000 --count 100 --seed-base 1000 --out-dir outputs/streams
#   GEN-PAIRS:  python kit_real.py gen-pairs --ckpt checkpoints/dna_prng.pt --real real_data_100k.txt --bits 1000000 --count 100 --seed-base 1000 --flip-bit 0 --out-dir outputs/pairs
#
# Notlar:
# - "bits" = hedef bit uzunluğu. 1 Mbit için --bits 1000000 (≈ 500k baz).
# - Çıktılar: .bits.txt (0/1), isteğe bağlı .dna.txt ve meta .json
# - Pairs modunda pairs.csv oluşturulur: her satır "pathA,pathB"

# dna_prng_real_core_random25k.py
# Her koşuda 25K veri rastgele seçilir (contiguous veya scattered),
# train=20K, val=5K. NO-REF tarzı profil/analiz ve güvenli çıktı yazımı.


import os, math, time, json, struct, hashlib, re, zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ====== Güvenli yol/dizin yardımcıları ======
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()  # IDE/Notebook için

def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

# === NVIDIA / CUDA ayarları ===
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
    except Exception:
        pass

# ====== Basit Konfig ======
BALANCE_GLOBAL   = True          # GC≈0.5 hedefini kota ile takip et
STRICT_QUOTA     = False         # <--- İSTENEN SENARYO: kapalı
SOFT_BALANCE     = True          # kota etkisini yumuşak uygula (logit tabanlı algoritmaya ek)
BALANCE_PERIOD   = 64
BALANCE_JITTER   = True
LAG1_DAMP        = 0.030         # kısa-erimli bağımlılık sönümleme
ALPHA_BASE       = 0.10          # kota bias başlangıcı
ALPHA_END        = 0.50          # kota bias son ağırlığı
PROMPT_SOURCE    = "uniform"     # başlangıç prompt’u
HOMOPOLYMER_MAX  = 5
USE_SHAKE        = False         # varsayılan: extractor yok (kullanıcı tercihi)
KS_BYTES         = 64 * 1024
KEY_BYTES        = 32

# --- 3-mer ve bigram yumuşatma (opsiyonel) ---
ENABLE_TRIMER_SMOOTH = True
TRIMER_ALPHA         = 0.26   # logit’e eklenecek katsayı (düşük/orta)
TRIMER_CLAMP         = 0.40   # |bias| ≤ bu sınır (güvenlik)
TRIMER_WARMUP        = 1280  # ilk adımlarda uygulama (kademeli başlatma)

ENABLE_BIGRAM_SMOOTH = True
BIGRAM_ALPHA         = 0.06
BIGRAM_CLAMP         = 0.25
BIGRAM_WARMUP        = 512

RULE_TEMP_ADAPTIVE   = 3.9    # kural seçimi sıcaklığı

# --- Kullanıcı dostu sabitler ---
REAL_PATH       = os.path.join(BASE_DIR, "yapaydna_1m.txt")  # Veri dosyan (FASTA/TXT)
TARGET_BITS     = 1_000_000                           # Üretilecek bit sayısı
EPOCHS          = 1
BATCH           = 256
LR              = 3e-4
OUT_DIR         = os.path.join(BASE_DIR, "outputs")
OUT_TAG         = "real_core_logit_constraints_trimer_bigram"

print("[OUT_DIR]", OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Opsiyonel istatistik paketleri ----
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

# --------------------------- Sabitler / Yardımcılar ---------------------------
DNA = ["A","C","G","T"]
VOCAB = {b:i for i,b in enumerate(DNA)}
INV_VOCAB = {i:b for b,i in VOCAB.items()}
SEED = 1337

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
ENC_RULE_NAMES = [f"R{i}: " + " ".join(f"{b}={bits}" for b,bits in rule.items()) for i,rule in enumerate(ENC_RULES)]

RULE_MATS = np.zeros((8,4,2), dtype=np.float32)
for r,rule in enumerate(ENC_RULES):
    for bch, bits in rule.items():
        b = VOCAB[bch]
        RULE_MATS[r,b,0] = 1.0 if bits[0] == '1' else 0.0
        RULE_MATS[r,b,1] = 1.0 if bits[1] == '1' else 0.0
RULE_MATS_T = torch.tensor(RULE_MATS)

# --------------------------- Dosya / Veri ---------------------------
def read_fasta_or_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    lines = [ln.strip() for ln in txt.splitlines() if ln and not ln.startswith('>')]
    seq = ''.join(lines).upper()
    return ''.join(ch for ch in seq if ch in VOCAB)

def make_generator(master_seed: int, device: str = 'cpu') -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(master_seed))
    return g

# --------------------------- 25K Örnekleme (yalnızca torch.Generator) ---------------------------
SAMPLING_MODE = "scattered"   # "contiguous" da seçebilirsin
SCATTER_CHUNK_LEN = 1000
SCATTER_NUM_CHUNKS = 25       # 25 × 1000 = 25K

def sample_25k_sequence(full: str, mode: str, gen: torch.Generator,
                        chunk_len: int = 1000, n_chunks: int = 25) -> str:
    need = 25_000
    N = len(full)
    assert N >= need, f"Kaynak çok kısa (N={N}) — en az 25K gerekir."

    if mode.lower() == "contiguous":
        s0 = int(torch.randint(0, N - need + 1, (1,), generator=gen).item())
        seg = full[s0:s0+need]
        print(f"[SAMPLE] contiguous window: start={s0}, len=25000")
        return seg

    assert chunk_len * n_chunks == need, "chunk_len * n_chunks 25,000 olmalı."
    grid_starts = torch.arange(0, N - chunk_len + 1, chunk_len)
    assert grid_starts.numel() >= n_chunks, "Kaynak uzunluğu bu grid ayarıyla yeterli değil."
    perm = torch.randperm(grid_starts.numel(), generator=gen)
    starts = grid_starts[perm[:n_chunks]].sort().values.tolist()
    parts = [full[int(s):int(s)+chunk_len] for s in starts]
    seg = "".join(parts)
    print(f"[SAMPLE] scattered: {n_chunks}×{chunk_len} parçadan toplandı (ilk start={int(starts[0])}, son start={int(starts[-1])})")
    return seg

# --------------------------- Dataset ---------------------------
class DNALMDataset(Dataset):
    def __init__(self, seq: str, ctx: int):
        self.tok = torch.tensor([VOCAB[b] for b in seq], dtype=torch.long)
        self.ctx = ctx
    def __len__(self):
        return max(0, len(self.tok)-self.ctx)
    def __getitem__(self, idx):
        x = self.tok[idx:idx+self.ctx]
        y = self.tok[idx+1:idx+self.ctx+1]
        return x, y

# --------------------------- Model ---------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 4
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.1
    lambda_rule: float = 0.2
    beta_uniform: float = 0.2
    tau_rule: float = 0.25
    tiny_rule_jitter: float = 1e-4

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
        self.drop_p  = cfg.dropout
        base = 10000.0
        d = self.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.cfg = cfg

    def _apply_rope(self, x, start_index: int = 0):
        B,H,T,D = x.shape
        x = x.view(B,H,T,D//2,2)
        x1, x2 = x[...,0], x[...,1]
        device = x1.device; dtype = x1.dtype
        pos = torch.arange(start_index, start_index + T, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('t,f->tf', pos, self.inv_freq)
        cos = freqs.cos().to(dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype)[None, None, :, :]
        xr0 = x1 * cos - x2 * sin
        xr1 = x1 * sin + x2 * cos
        return torch.stack([xr0, xr1], dim=-1).reshape(B,H,T,D)

    def forward(self, x):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim
        k = self.key(x).view(B, T, H, D).transpose(1, 2)
        q = self.query(x).view(B, T, H, D).transpose(1, 2)
        v = self.value(x).view(B, T, H, D).transpose(1, 2)
        q = self._apply_rope(q, start_index=0)
        k = self._apply_rope(k, start_index=0)
        attn = F.scaled_dot_product_attention(q, k, v, None, self.drop_p if self.training else 0.0, True)
        y = attn.transpose(1,2).contiguous().view(B, T, H*D)
        return self.proj(y)

    def forward_with_past(self, x, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim
        k = self.key(x).view(B, T, H, D).transpose(1, 2)
        q = self.query(x).view(B, T, H, D).transpose(1, 2)
        v = self.value(x).view(B, T, H, D).transpose(1, 2)
        S = 0 if (past_kv is None) else past_kv[0].size(2)
        q = self._apply_rope(q, start_index=S)
        k = self._apply_rope(k, start_index=S)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v, None, self.drop_p if self.training else 0.0, (past_kv is None) or (T > 1))
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
            nn.Dropout(cfg.dropout),
        )
        self.cfg = cfg

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_with_past(self, x, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        y, present = self.attn.forward_with_past(self.ln1(x), past_kv=past_kv)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, present

class MiniGPTDualHead(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.base_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.rule_head = nn.Linear(cfg.n_embd, 8, bias=False)

    def forward(self, idx, targets=None):
        B,T = idx.size()
        assert T <= self.cfg.block_size
        x = self.tok_emb(idx); x = self.drop(x)
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        base_logits = self.base_head(x)
        rule_logits = self.rule_head(x)
        loss = None
        if targets is not None:
            base_loss = nn.functional.cross_entropy(base_logits.view(-1,4), targets.view(-1))
            with torch.no_grad():
                p_base = torch.softmax(base_logits, dim=-1)       # [B,T,4]
                RM = RULE_MATS_T.to(p_base.device)                # [8,4,2]
                Eb = torch.einsum('btk,rkj->btrj', p_base, RM)    # [B,T,8,2]
                score = 1.0 - (torch.abs(Eb[...,0]-0.5) + torch.abs(Eb[...,1]-0.5))/2.0
                score = score + self.cfg.tiny_rule_jitter * torch.randn_like(score)
                q_rule = torch.softmax(score / self.cfg.tau_rule, dim=-1)
            logp_rule = torch.log_softmax(rule_logits, dim=-1)
            rule_ce = -(q_rule * logp_rule).sum(dim=-1).mean()
            p_rule = torch.softmax(rule_logits, dim=-1)
            unif = torch.full_like(p_rule, 1.0/8.0)
            kl = (p_rule * (torch.log(p_rule+1e-12) - torch.log(unif))).sum(dim=-1).mean()
            loss = base_loss + self.cfg.lambda_rule*rule_ce + self.cfg.beta_uniform*kl
        return base_logits, rule_logits, loss

    @torch.no_grad()
    def forward_with_past(self, idx, past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        B,T = idx.size()
        x = self.tok_emb(idx); x = self.drop(x)
        presents: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, blk in enumerate(self.blocks):
            pkv_i = None if past_kv is None else past_kv[i]
            x, present_i = blk.forward_with_past(x, past_kv=pkv_i)
            presents.append(present_i)
        x = self.ln_f(x)
        base_logits = self.base_head(x)
        rule_logits = self.rule_head(x)
        return base_logits, rule_logits, presents

# --------------------------- Eğitim ---------------------------
@dataclass
class TrainOut:
    model: MiniGPTDualHead
    val_bpb: float

def bits_per_base(model: MiniGPTDualHead, val_loader: DataLoader, device: str) -> float:
    model.eval(); nll, n = 0.0, 0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            base_logits, _, _ = model(xb, yb)
            loss = nn.functional.cross_entropy(base_logits.view(-1,4), yb.view(-1), reduction='sum')
            nll += loss.item(); n += yb.numel()
    return float((nll/n)/math.log(2))

def train_model(train_seq: str, val_seq: str, cfg: GPTConfig,
                epochs: int=1, bs: int=256, lr: float=3e-4, device: Optional[str]=None) -> TrainOut:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniGPTDualHead(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_ds = DNALMDataset(train_seq, cfg.block_size)
    val_ds   = DNALMDataset(val_seq,   cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=True, num_workers=0)

    best_bpb = 9e9
    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train ep{ep+1}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            _, _, loss = model(xb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        bpb = bits_per_base(model, val_loader, device)
        best_bpb = min(best_bpb, bpb)
        print(f"val bits/base: {bpb:.4f} (best {best_bpb:.4f})")
    return TrainOut(model, best_bpb)

# --------------------------- Üretim Yardımcıları ---------------------------
def _compute_target_probs(gc_target: Optional[float], pA: Optional[float], pC: Optional[float], pG: Optional[float], pT: Optional[float]) -> Dict[str,float]:
    if gc_target is not None:
        assert 0.0 <= gc_target <= 1.0
        pC = pG = gc_target/2.0
        pA = pT = (1.0-gc_target)/2.0
    else:
        assert None not in (pA,pC,pG,pT)
    probs = {"A":float(pA),"C":float(pC),"G":float(pG),"T":float(pT)}
    s = sum(probs.values())
    for k in probs: probs[k] /= s
    return probs

# ---- LOGIT KISITLARI + 3-MER/BIGRAM YUMUŞATMA ----
@torch.no_grad()
def generate_constrained_with_rules(
    model: MiniGPTDualHead,
    start_tokens: torch.LongTensor,
    out_len: int,
    temperature: float=1.0,
    top_k: Optional[int]=None,
    rule_temp: float=RULE_TEMP_ADAPTIVE,
    device: str='cpu',
    gc_target: float = 0.5,
    homopolymer_max: Optional[int] = None,
    balance_mode: str = 'none',
    rule_mode: str = 'adaptive',
    rule_id: int = 0,
    gen: Optional[torch.Generator] = None
) -> Tuple[str, List[int], str]:

    model.eval()
    cfg = model.cfg

    # Rastgele üreteç
    if gen is None:
        run_seed = int.from_bytes(os.urandom(8), 'little') & ((1<<63)-1)
        seed_env = os.getenv("RUN_SEED")
        if seed_env and seed_env.strip():
            try:
                run_seed = int(seed_env, 0)
                print(f"[RUN] run_seed(override)={run_seed}")
            except Exception as e:
                print(f"[WARN] RUN_SEED parse failed: {e}")
        gen = make_generator(run_seed, device=device)
        print(f"[RUN] run_seed={run_seed}")

    # Başlangıç dizisi
    idx = start_tokens.clone().to(device)
    rules: List[int] = []
    bits_out: List[str] = []

    # Hedef dağılım ve başlangıç kotaları
    target = _compute_target_probs(gc_target, None,None,None,None) if balance_mode!='none' else None
    if balance_mode=='global':
        quotas = {b:int(round(out_len*target[b])) for b in DNA}
        diff = out_len - sum(quotas.values())
        if diff != 0:
            order_idx = torch.randperm(4, generator=gen, device=torch.device(device)).tolist()
            order = [DNA[i] for i in order_idx]
            for b in order:
                if diff == 0: break
                quotas[b] += 1 if diff > 0 else -1; diff += -1 if diff > 0 else 1
        remain = quotas.copy()
    else:
        remain = {b:10**18 for b in DNA}

    # Sayaçlar
    counts = {b:0 for b in DNA}
    last_base: Optional[int] = None
    run_len = 0

    # 3-mer / 2-mer sayaçları
    tri_counts = torch.zeros(64, dtype=torch.long, device=device)
    bi_counts  = torch.zeros(16, dtype=torch.long, device=device)
    n_tri = 0
    n_bi  = 0
    prev1: Optional[int] = None
    prev2: Optional[int] = None

    base_logits, rule_logits, past = model.forward_with_past(idx, past_kv=None)

    EPS_MIN   = 1e-6

    for step in tqdm(range(out_len), desc='generate'):
        base_step = base_logits[:, -1, :].clone() / max(1e-4, temperature)
        rule_step = rule_logits[:, -1, :].clone() / max(1e-4, rule_temp)

        # 1) Log-softmax taban
        log_p = torch.log_softmax(base_step, dim=-1)[0]

        # 2) Kota-bias (logit): remain/target oranının log'u
        t_emitted = int(idx.size(1) - start_tokens.size(1))
        steps_left = max(1, out_len - t_emitted)
        if balance_mode != 'none' and target is not None:
            target_vec = torch.tensor([target['A'],target['C'],target['G'],target['T']], device=log_p.device, dtype=log_p.dtype)
            rem_vec = torch.tensor([
                max(0, remain['A'])/steps_left,
                max(0, remain['C'])/steps_left,
                max(0, remain['G'])/steps_left,
                max(0, remain['T'])/steps_left,
            ], device=log_p.device, dtype=log_p.dtype)
            ratio = torch.clamp(rem_vec/torch.clamp(target_vec, EPS_MIN), min=EPS_MIN)
            prog  = t_emitted / max(1, out_len-1)
            alpha = ALPHA_BASE + (ALPHA_END-ALPHA_BASE)*prog
            alpha = float(max(0.0, min(2.0, alpha)))
            log_p = log_p + alpha * torch.log(ratio)

        # 3) Lag-1 sönüm (logit)
        if last_base is not None and LAG1_DAMP > 0.0:
            log_p[last_base] += math.log(max(EPS_MIN, 1.0 - LAG1_DAMP))

        # 4) 3-mer / bigram yumuşatma (logit) — bağlama bağlı düzeltme
        #    prev2, prev1 mevcutsa: her aday b için oluşacak (prev2,prev1,b) üçlüsünün
        #    sayımının beklenen değere (≈ n_tri/64) yakınlığına göre bias uygula.
        if ENABLE_TRIMER_SMOOTH and (prev2 is not None) and (prev1 is not None) and (t_emitted >= 2 + TRIMER_WARMUP):
            exp_tri = max(1.0, n_tri / 64.0)
            tri_bias = torch.zeros(4, dtype=log_p.dtype, device=log_p.device)
            for b in range(4):
                tri_idx = (prev2 << 4) | (prev1 << 2) | b
                c = float(tri_counts[tri_idx].item())
                ratio = (c + 1.0) / (exp_tri + 1.0)
                bias  = -TRIMER_ALPHA * math.log(max(EPS_MIN, ratio))
                bias  = float(max(-TRIMER_CLAMP, min(TRIMER_CLAMP, bias)))
                tri_bias[b] = bias
            log_p = log_p + tri_bias

        if ENABLE_BIGRAM_SMOOTH and (prev1 is not None) and (t_emitted >= 1 + BIGRAM_WARMUP):
            exp_bi = max(1.0, n_bi / 16.0)
            bi_bias = torch.zeros(4, dtype=log_p.dtype, device=log_p.device)
            for b in range(4):
                bi_idx = (prev1 << 2) | b
                c = float(bi_counts[bi_idx].item())
                ratio = (c + 1.0) / (exp_bi + 1.0)
                bias  = -BIGRAM_ALPHA * math.log(max(EPS_MIN, ratio))
                bias  = float(max(-BIGRAM_CLAMP, min(BIGRAM_CLAMP, bias)))
                bi_bias[b] = bias
            log_p = log_p + bi_bias

        # 5) Homopolimer kontrol (sert)
        if homopolymer_max and last_base is not None and run_len >= homopolymer_max:
            log_p[last_base] = -1e9

        # 6) top-k (opsiyonel)
        if top_k is not None:
            k = min(top_k, 4)
            _, idxk = torch.topk(log_p, k=k)
            keep = torch.zeros(4, dtype=torch.bool, device=log_p.device); keep[idxk] = True
            if homopolymer_max and last_base is not None and run_len >= homopolymer_max:
                keep[last_base] = False
            floor_log = math.log(EPS_MIN)
            for i in range(4):
                if not keep[i]:
                    log_p[i] = floor_log

        # 7) Olasılıklar
        p_base = torch.softmax(log_p, dim=-1)
        if (not torch.isfinite(p_base).all()) or float(p_base.sum()) <= 0.0:
            log_p = torch.log_softmax(base_step, dim=-1)[0]
            if homopolymer_max and last_base is not None and run_len >= homopolymer_max:
                log_p[last_base] = -1e9
            p_base = torch.softmax(log_p, dim=-1)

        # 8) Numune al
        next_base = torch.multinomial(p_base, num_samples=1, generator=gen)
        b = int(next_base.item())

        # 9) Kural seçimi
        if rule_mode == 'learned':
            p_rule = torch.softmax(rule_step[0], dim=-1)
            r = int(torch.multinomial(p_rule, num_samples=1, generator=gen).item())
        elif rule_mode == 'uniform':
            r = int(torch.randint(0, 8, (1,), generator=gen, device=p_base.device).item())
        elif rule_mode == 'adaptive':
            RM = RULE_MATS_T.to(p_base.device)          # [8,4,2]
            Eb = torch.einsum('f,rfj->rj', p_base, RM)  # [8,2]
            score = 1.0 - (torch.abs(Eb[:, 0] - 0.5) + torch.abs(Eb[:, 1] - 0.5)) / 2.0
            pr = torch.softmax(score / max(1e-4, rule_temp), dim=-1)
            r = int(torch.multinomial(pr, 1, generator=gen).item())
        else:
            r = int(max(0, min(7, rule_id)))

        base_char = INV_VOCAB[b]
        rules.append(r)
        bits_out.append(ENC_RULES[r][base_char])

        # Sayaçlar ve kalan kota
        counts[base_char] += 1
        if balance_mode=='global':
            remain[base_char] -= 1

        # N-gram sayaçlarını güncelle
        if prev1 is not None:
            bi_idx = (prev1 << 2) | b
            bi_counts[bi_idx] += 1
            n_bi += 1
        if (prev2 is not None) and (prev1 is not None):
            tri_idx = (prev2 << 4) | (prev1 << 2) | b
            tri_counts[tri_idx] += 1
            n_tri += 1

        # Run-length ve prev güncelle
        if last_base is None or b != last_base:
            last_base = b; run_len = 1
        else:
            run_len += 1
        prev2 = prev1
        prev1 = b

        # Modeli ileri al ve KV kırp
        tok = torch.tensor([[b]], dtype=torch.long, device=device)
        base_logits, rule_logits, past = model.forward_with_past(tok, past_kv=past)
        for i in range(len(past)):
            K, V = past[i]
            if K.size(2) > cfg.block_size:
                past[i] = (K[:, :, -cfg.block_size:, :].contiguous(),
                           V[:, :, -cfg.block_size:, :].contiguous())
        idx = torch.cat([idx, tok], dim=1)

    dna_seq = ''.join(INV_VOCAB[int(t)] for t in idx[0, start_tokens.size(1):])
    bit_seq = ''.join(bits_out)
    return dna_seq, rules, bit_seq

# --------------------------- Analiz / meta ---------------------------
def _entropy_bits_per_base(counts: Dict[str,int]) -> float:
    N = sum(counts.values())
    if N == 0: return 0.0
    ps = [counts.get(b,0)/N for b in 'ACGT']
    return float(-sum(p*math.log(p,2) for p in ps if p>0))

def _analyze(dna_seq: str, bits: str) -> Dict[str,object]:
    N = len(dna_seq)
    counts = {b: dna_seq.count(b) for b in 'ACGT'}
    ent = _entropy_bits_per_base(counts)
    ones = bits.count('1'); n_bits = len(bits)
    p_mono = float(binomtest(ones, n_bits, p=0.5).pvalue) if N>0 else 1.0
    exp = [N/4.0]*4
    p_chi = float(chisquare([counts['A'],counts['C'],counts['G'],counts['T']], exp).pvalue) if N>0 else 1.0
    return {"len_bases": N, "entropy_bits_per_base": ent, "monobit_p": p_mono, "chi2_p": p_chi, "counts": counts}

# --- 3-mer özeti (doğrulama için) ---
def _kmer3_summary(seq: str) -> Dict[str, float]:
    from collections import Counter
    if len(seq) < 3:
        return {"n_windows": 0, "exp_per_kmer": 0.0, "min": 0, "median": 0.0, "max": 0, "mean": 0.0, "chi2_p": 1.0}
    n = len(seq) - 2
    all3 = [a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']
    cnt = Counter(seq[i:i+3] for i in range(n))
    obs = [cnt.get(tri, 0) for tri in all3]
    exp = [n/64.0] * 64
    arr = np.array(obs, dtype=float)
    med = float(np.median(arr))
    chi_p = float(chisquare(obs, exp).pvalue) if _HAVE_SCIPY else 1.0
    return {"n_windows": int(n), "exp_per_kmer": float(n/64.0), "min": int(arr.min()),
            "median": med, "max": int(arr.max()), "mean": float(arr.mean()), "chi2_p": chi_p}

# Bit paketleme ve keystream (USE_SHAKE=False varsayılan)
def _bits_to_bytes(bit_str: str) -> bytes:
    if not bit_str: return b""
    pad = (-len(bit_str)) % 8
    if pad: bit_str += '0' * pad
    return int(bit_str, 2).to_bytes(len(bit_str)//8, 'big')

def keystream_from_transformer(bits: str, run_seed: int, n_bytes: int,
                               domain: bytes = b"DNA-PRNG/core-v1",
                               model_fingerprint: Optional[bytes] = None) -> bytes:
    shake = hashlib.shake_256()
    shake.update(domain + b"\x00")
    shake.update(struct.pack(">Q", int(run_seed)))
    if model_fingerprint:
        shake.update(b"\x01" + model_fingerprint)
    shake.update(b"\x02" + struct.pack(">Q", len(bits)))
    shake.update(b"\x03" + _bits_to_bytes(bits))
    return shake.digest(n_bytes)

def pack_bits_to_bytes(bitstr: str) -> bytes:
    out = bytearray(); acc = 0; nbits = 0
    for ch in bitstr:
        acc = (acc << 1) | (1 if ch == '1' else 0); nbits += 1
        if nbits == 8: out.append(acc); acc = 0; nbits = 0
    if nbits: out.append(acc << (8 - nbits))
    return bytes(out)

# --------------------------- Ana akış ---------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = GPTConfig()

    # 0) RUN_SEED: örnekleme ve üretim için tek kaynak
    run_seed = int.from_bytes(os.urandom(8), 'little') & ((1<<63)-1)
    seed_env = os.getenv("RUN_SEED")
    if seed_env and seed_env.strip():
        try:
            run_seed = int(seed_env, 0)
            print(f"[RUN] run_seed(override)={run_seed}")
        except Exception as e:
            print(f"[WARN] RUN_SEED parse failed: {e}")
    torch_gen = make_generator(run_seed, device=device)
    print(f"[RUN] run_seed={run_seed}")

    # 1) Veri oku
    full = read_fasta_or_txt(REAL_PATH)
    assert len(full) >= 100_000, f"Dosya çok kısa: {REAL_PATH} (>=100K olmalı)"

    # 2) 25K örnekleme (yalnızca torch_gen)
    picked25k = sample_25k_sequence(full, SAMPLING_MODE, torch_gen,
                                chunk_len=SCATTER_CHUNK_LEN, n_chunks=SCATTER_NUM_CHUNKS)

    # Train/Val böl
    train = picked25k[:20_000]
    val   = picked25k[20_000:25_000]
    print(f"[SPLIT] train=20K, val=5K (sampling_mode={SAMPLING_MODE})")

    # 3) Eğit
    to = train_model(train, val, cfg, epochs=EPOCHS, bs=BATCH, lr=LR, device=device)

    # 4) Başlangıç prompt (uniform)
    def _choose_start_tokens(cfg: GPTConfig) -> torch.LongTensor:
        # Tamamen torch_gen ile uniform başlat
        return torch.randint(0, 4, (1, cfg.block_size), generator=torch_gen, device=torch.device(device)).long().cpu()

    start = _choose_start_tokens(cfg)

    # 5) Çıkış uzunluğu (baz) – bit sayısına göre
    def _ensure_even_bits(nbits: int) -> int:
        return int(nbits + (nbits & 1))
    out_len_bases = _ensure_even_bits(TARGET_BITS) // 2

    # 6) Performans profili (I/O hariç)
    if _HAVE_PSUTIL:
        proc = psutil.Process(os.getpid())
        cpu0_user, cpu0_sys = proc.cpu_times().user, proc.cpu_times().system
        rss0 = proc.memory_info().rss
    else:
        cpu0_fallback = time.process_time()
        cpu0_user = cpu0_sys = 0.0
        rss0 = None
    t0_ns = time.perf_counter_ns()

    # 7) Üret – LOGIT + 3-MER/BIGRAM DENGELEME ile
    dna, rules, bits = generate_constrained_with_rules(
        to.model, start, out_len_bases,
        temperature=1.0, top_k=None, rule_temp=RULE_TEMP_ADAPTIVE,
        device=device, gc_target=0.5, homopolymer_max=HOMOPOLYMER_MAX,
        balance_mode=('global' if BALANCE_GLOBAL else 'none'),
        rule_mode='adaptive', rule_id=0, gen=torch_gen
    )
    if len(bits) > TARGET_BITS:
        bits = bits[:TARGET_BITS]

    # 8) Profil bitişi
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
    eff = (len(dna) / t_core_ms) if t_core_ms > 0 else float('inf')
    perf = {"t_core_ms": t_core_ms, "cpu_s": cpu_s, "ram_mb": ram_mb,
            "eff_char_per_ms": eff, "n_char": len(dna)}

    # 9) Özet yazdır
    print("\n--- CORE PERFORMANCE (no I/O / no analysis) ---")
    ram_str = f"{perf['ram_mb']:.2f} MB" if perf['ram_mb'] is not None else "N/A"
    print(f"Sure(ms)={perf['t_core_ms']:.3f}  CPU(s)={perf['cpu_s']:.3f}  RAM_delta={ram_str}  "
          f"Verimlilik={perf['eff_char_per_ms']:.2f} char/ms")

    from collections import Counter
    rcounts = Counter(rules)
    top_r = max(range(8), key=lambda k: rcounts.get(k, 0))
    mapping_str = " ".join([f"{b}={ENC_RULES[top_r][b]}" for b in ['A','C','G','T']])
    print(f"\n[RULE] most used: R{top_r} -> {ENC_RULE_NAMES[top_r]}")
    print(f"[RULE] mapping: {mapping_str}")

    def _analyze_and_print(dna_seq: str, bits_str: str):
        stats = _analyze(dna_seq, bits_str)
        print("\n--- ANALYSIS ---")
        print(f"bases={stats['len_bases']:,}  bits={len(bits_str):,}  entropy(bits/base)={stats['entropy_bits_per_base']:.6f}")
        print(f"mono_p={stats['monobit_p']:.6g}  chi_p={stats['chi2_p']:.6g}")
        c = stats['counts']; tot = stats['len_bases']
        if tot > 0:
            print(f"A={c['A']} ({100*c['A']/tot:.2f}%)  C={c['C']} ({100*c['C']/tot:.2f}%)  "
                  f"G={c['G']} ({100*c['G']/tot:.2f}%)  T={c['T']} ({100*c['T']/tot:.2f}%)")
        return stats

    stats = _analyze_and_print(dna, bits)

    def _gc_summary_print(tag, seq):
        N = len(seq)
        gc = seq.count('G') + seq.count('C')
        p_hat = gc / max(1, N)
        se_emp = math.sqrt(max(p_hat * (1.0 - p_hat) / max(1,N), 1e-12))
        se_the = math.sqrt(max(0.5 * 0.5 / max(1,N), 1e-12))
        z = 1.96
        print(f"\n--- GC SUMMARY ({tag}) ---")
        print(f"N={N:,}  GC={p_hat:.5f}  95%CI_emp(+/-)={z*se_emp:.5f}  (ideal p=0.5 -> 95%CI(+/-)={z*se_the:.5f})")

    def _k3_print(tag, seq):
        if len(seq) < 3:
            print(f"\n--- 3-MER COVERAGE ({tag}) ---\nveri kisa"); return
        n = len(seq) - 2
        from collections import Counter as C2
        all3 = [a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']
        cnt = C2(seq[i:i+3] for i in range(n))
        obs = [cnt.get(tri, 0) for tri in all3]
        exp = [n/64.0] * 64
        arr = np.array(obs, dtype=float)
        med = float(np.median(arr))
        chi_p = float(chisquare(obs, exp).pvalue) if _HAVE_SCIPY else 1.0
        print(f"\n--- 3-MER COVERAGE ({tag}) ---")
        print(f"windows={n:,}  expected_per_3mer~{n/64.0:.2f}  "
              f"observed min/median/max = {int(arr.min())}/{med:.1f}/{int(arr.max())}  "
              f"chi2_p={chi_p:.6g}")

    _gc_summary_print("Generated", dna)
    _k3_print("Generated", dna)

    # 10) Diğer özetler
    print("\n--- RULE USAGE ---")
    for i in range(8):
        print(f"R{i}: {rcounts.get(i,0)}")

    print('\nbit_len =', len(bits))
    print('p(1) =', (bits.count('1')/len(bits)) if len(bits)>0 else 0.0)

    if len(dna) > 0:
        max_run = max(len(m.group(0)) for m in re.finditer(r'(A+|C+|G+|T+)', dna))
    else:
        max_run = 0
    print('max homopolymer run =', max_run)

    comp = zlib.compress(bits.encode('ascii'), level=9) if len(bits)>0 else b""
    print('compression ratio =', (len(comp)/len(bits)) if len(bits)>0 else 0.0)

    # 11) Opsiyonel keystream + 256-bit key dosyaları (USE_SHAKE=False default)
    ts = time.strftime('%Y%m%d_%H%M%S')
    if USE_SHAKE:
        keystream = keystream_from_transformer(bits, run_seed, KS_BYTES,
                                               domain=b"DNA-PRNG/core-v1",
                                               model_fingerprint=None)
        key_256 = keystream_from_transformer(bits, run_seed, KEY_BYTES,
                                             domain=b"DNA-PRNG/key-v1",
                                             model_fingerprint=None)
    else:
        raw = pack_bits_to_bytes(bits)
        keystream = raw[:KS_BYTES]
        key_256 = raw[:KEY_BYTES]

    ks_path = os.path.join(OUT_DIR, f'keystream_{ts}.bin')
    _ensure_parent(ks_path)
    with open(ks_path, 'wb') as f:
        f.write(keystream)
    print(f"\n[KEYSTREAM] {len(keystream)} bytes yazildi -> {ks_path}")
    print(f"[KEYSTREAM] ilk 32 bayt (hex): {keystream[:32].hex()}")

    key_path = os.path.join(OUT_DIR, f'key_{ts}.hex')
    _ensure_parent(key_path)
    with open(key_path, 'w', encoding='utf-8') as f:
        f.write(key_256.hex() + "\n")
    print(f"[KEY] {8*len(key_256)}-bit (hex): {key_256.hex()}")
    print(f"[KEY SAVED] {key_path}")

    # 12) Kaydet (bits, rules, dna, meta)
    base = os.path.join(OUT_DIR, f"{OUT_TAG}_{ts}")
    bits_path  = base + ".bits.txt"
    rules_path = base + ".rules.txt"
    dna_path   = base + ".dna.txt"
    meta_path  = base + ".json"

    for p in (bits_path, rules_path, dna_path, meta_path):
        _ensure_parent(p)

    with open(bits_path, 'w', encoding='utf-8') as f:
        f.write(bits)
    with open(rules_path, 'w', encoding='utf-8') as f:
        f.write(" ".join(str(r) for r in rules))
    with open(dna_path, 'w', encoding='utf-8') as f:
        f.write(dna)

    meta = {
        "mode": "scenario_real_core_logit_constraints+trimer_bigram",
        "trained_now": True,
        "run_seed": int(run_seed),
        "val_bits_per_base": to.val_bpb,
        "bits_len": len(bits),
        "rule_counts": {str(k): rcounts.get(k, 0) for k in range(8)},
        "top_rule": int(top_r),
        "top_rule_name": ENC_RULE_NAMES[top_r],
        "top_rule_mapping": {b: ENC_RULES[top_r][b] for b in ['A','C','G','T']},
        "sampling": {
            "mode": "scattered",
            "scatter_chunk_len": 1000,
            "scatter_num_chunks": 25
        },
        "smoothing": {
            "enable_trimer": ENABLE_TRIMER_SMOOTH,
            "trimer_alpha": TRIMER_ALPHA,
            "trimer_clamp": TRIMER_CLAMP,
            "trimer_warmup": TRIMER_WARMUP,
            "enable_bigram": ENABLE_BIGRAM_SMOOTH,
            "bigram_alpha": BIGRAM_ALPHA,
            "bigram_clamp": BIGRAM_CLAMP,
            "bigram_warmup": BIGRAM_WARMUP
        },
        "perf": {
            "t_core_ms": perf["t_core_ms"],
            "cpu_s":     perf["cpu_s"],
            "ram_mb":    perf["ram_mb"],
            "eff_char_per_ms": perf["eff_char_per_ms"],
            "n_char":    perf["n_char"]
        }
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVED] dna  -> {dna_path}")
    print(f"[SAVED] bits -> {bits_path}")
    print(f"[SAVED] rules-> {rules_path}")
    print(f"[SAVED] meta -> {meta_path}")

    # 13) Sabit adlar (opsiyonel)
    dna_fixed  = os.path.join(OUT_DIR, "sentetic_DNA_500k.txt")
    bits_fixed = os.path.join(OUT_DIR, "sentetic_1Mbits.txt")
    _ensure_parent(dna_fixed); _ensure_parent(bits_fixed)
    with open(dna_fixed, 'w', encoding='utf-8') as f:
        f.write(dna[:500_000])
    with open(bits_fixed,'w', encoding='utf-8') as f:
        f.write(bits[:1_000_000])
    print(f"[SAVED] fixed names -> {dna_fixed}, {bits_fixed}")
