"""Neural-network building blocks for the frequency x severity model (kind='nn' in freqsev.FreqSev).

Architecture (CANN-style, Schelldorfer & Wuthrich): categorical embeddings + standardised numerics feed (a) a small MLP and (b) a plain linear
"skip" layer, summed on the log scale. The skip layer is the GLM part, the MLP learns the departures from it.
* FreqNet: one network for all perils, one Poisson head per peril, log(exposure x completeness) offset, bias initialised at log(base rate).
* SevNet: one network for all perils, one Gamma head per peril (log link), trained on trended, capped claim costs.
Early stopping uses a customer-grouped hold-out of the training rows (never the validation fold).
Options: tabm_k > 0 turns each MLP into a TabM-style BatchEnsemble (k rank-1-modulated members sharing weights; trained member-wise, averaged
on the log scale at prediction); entity_cols adds high-cardinality id columns (customer, address) as small learned embeddings = a random-effect
per entity (rare/unseen entities share index 0); per-row weights (recency / domain shift) enter the losses.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import SEED


class NNX:
    """Encoded design matrix: standardised numerics (float32) and integer category codes (0 = unseen/rare/missing)."""

    def __init__(self, num: np.ndarray, cat: np.ndarray):
        self.num, self.cat = num, cat

    def __getitem__(self, idx):
        return NNX(self.num[idx], self.cat[idx])

    def __len__(self):
        return len(self.num)


class Encoder:
    def __init__(self, num_cols, cat_cols, min_level_count: int = 200, entity_cols=(), entity_min_count: int = 2, entity_dim: int = 4):
        self.entity_cols, self.entity_min_count, self.entity_dim = list(entity_cols), entity_min_count, entity_dim
        self.num_cols, self.cat_cols, self.min_level_count = list(num_cols), list(cat_cols) + self.entity_cols, min_level_count

    def fit(self, df: pd.DataFrame):
        num = df[self.num_cols].astype("float32")
        self.lo_, self.hi_ = num.quantile(0.005), num.quantile(0.995)
        num = num.clip(self.lo_, self.hi_, axis=1)
        self.med_ = num.median()
        self.miss_cols_ = [c for c in self.num_cols if df[c].isna().any()]
        filled = num.fillna(self.med_)
        self.mu_, self.sd_ = filled.mean(), filled.std().replace(0, 1.0).fillna(1.0)
        self.levels_ = {}
        for c in self.cat_cols:
            vc = df[c].astype(object).where(df[c].notna(), "missing").astype(str).value_counts()
            floor = self.entity_min_count if c in self.entity_cols else self.min_level_count
            self.levels_[c] = {lv: i + 1 for i, lv in enumerate(vc[vc >= floor].index)}
        self.cards_ = [len(v) + 1 for v in self.levels_.values()]
        self.emb_dims_ = [self.entity_dim if c in self.entity_cols else min(8, (len(v) + 1) // 2 + 1) for c, v in self.levels_.items()]
        self.n_num_ = len(self.num_cols) + len(self.miss_cols_)
        return self

    def transform(self, df: pd.DataFrame) -> NNX:
        raw = df[self.num_cols].astype("float32")
        num = raw.clip(self.lo_, self.hi_, axis=1)
        z = ((num.fillna(self.med_) - self.mu_) / self.sd_).clip(-6.0, 6.0).to_numpy(dtype="float32")
        if self.miss_cols_:
            z = np.hstack([z, raw[self.miss_cols_].isna().to_numpy(dtype="float32")])
        cat = np.column_stack([
            df[c].astype(object).where(df[c].notna(), "missing").astype(str).map(self.levels_[c]).fillna(0).to_numpy(dtype="int64")
            for c in self.cat_cols]) if self.cat_cols else np.zeros((len(df), 0), dtype="int64")
        return NNX(z, cat)


class _BELinear(nn.Module):
    """BatchEnsemble linear layer: k members share W and differ by rank-1 input/output scalings and their own bias (TabM)."""

    def __init__(self, d_in: int, d_out: int, k: int, first: bool):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.r = nn.Parameter(torch.randn(k, d_in).sign() if first else torch.ones(k, d_in))   # random-sign input scalings decorrelate the members
        self.s = nn.Parameter(torch.ones(k, d_out))
        self.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):   # (B, k, d_in) -> (B, k, d_out)
        return self.lin(x * self.r) * self.s + self.b


class _Net(nn.Module):
    """Embeddings + MLP + linear skip. tabm_k > 0: the MLP and head are BatchEnsembles; the output is (B, k, out) instead of (B, out)."""

    def __init__(self, n_num: int, cards, out_dim: int, hidden=(192, 96), dropout: float = 0.1, emb_max: int = 8, emb_dims=None, tabm_k: int = 0):
        super().__init__()
        dims = emb_dims if emb_dims is not None else [min(emb_max, c // 2 + 1) for c in cards]
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, dims)])
        d_in = n_num + sum(dims)
        self.k = tabm_k
        layers, d = [], d_in
        for li, h in enumerate(hidden):
            layers += [_BELinear(d, h, tabm_k, first=(li == 0)) if tabm_k else nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.head = _BELinear(d, out_dim, tabm_k, first=False) if tabm_k else nn.Linear(d, out_dim)
        self.skip = nn.Linear(d_in, out_dim)
        for m in (self.head.lin if tabm_k else self.head, self.skip):
            nn.init.zeros_(m.weight)
        for b in ((self.head.b,) if tabm_k else (self.head.bias,)) + (self.skip.bias,):
            nn.init.zeros_(b)

    def forward(self, num, cat):
        z = torch.cat([num] + [e(cat[:, i]) for i, e in enumerate(self.embs)], dim=1) if len(self.embs) else num
        if self.k:
            return self.head(self.mlp(z[:, None, :].expand(-1, self.k, -1))) + self.skip(z)[:, None, :]
        return self.head(self.mlp(z)) + self.skip(z)


def _train(net, loss_fn, tensors, val_mask, epochs, lr, batch, wd, patience, seed, threads):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    n = len(tensors[0])
    tr_idx, va_idx = torch.from_numpy(np.nonzero(~val_mask)[0]), torch.from_numpy(np.nonzero(val_mask)[0])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for g in opt.param_groups:
            g["lr"] = lr * (0.8 ** ep)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(net, *[t[b] for t in tensors]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net, *[t[va_idx] for t in tensors])) if len(va_idx) else float(ep)
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _predict(net, X: NNX, batch: int = 65536) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            o = net(torch.from_numpy(X.num[i:i + batch]), torch.from_numpy(X.cat[i:i + batch]))
            out.append((o.mean(1) if o.dim() == 3 else o).numpy())   # TabM: members are averaged on the log scale
    return np.vstack(out)


class FreqNet:
    """Poisson heads for K perils. counts (n, K), offset_expo (n, K) = exposure x completeness."""

    def __init__(self, enc: Encoder, hidden=(192, 96), dropout=0.1, epochs=12, lr=2e-3, batch=2048, wd=1e-4, patience=2, n_seeds=1, threads=4, tabm_k=0):
        self.enc, self.hidden, self.dropout, self.epochs, self.lr, self.batch, self.wd = enc, hidden, dropout, epochs, lr, batch, wd
        self.patience, self.n_seeds, self.threads, self.tabm_k = patience, n_seeds, threads, tabm_k

    def fit(self, X: NNX, counts: np.ndarray, expo: np.ndarray, val_mask: np.ndarray, w: np.ndarray | None = None):
        self.base_ = counts.sum(0) / expo.sum(0)
        b0 = torch.tensor(np.log(self.base_), dtype=torch.float32)
        tensors = [torch.from_numpy(X.num), torch.from_numpy(X.cat), torch.from_numpy(counts.astype("float32")),
                   torch.from_numpy(np.log(np.maximum(expo, 1e-6)).astype("float32")),
                   torch.from_numpy((np.ones(len(counts)) if w is None else w).astype("float32"))]

        def loss(net, num, cat, n, logoff, ww):
            out = net(num, cat)
            eta = (out if out.dim() == 3 else out[:, None, :]) + b0 + logoff[:, None, :]
            per_row = (torch.exp(eta) - n[:, None, :] * eta).sum(2).mean(1)   # mean over TabM members
            return (per_row * ww).sum() / ww.sum()

        self.nets_ = []
        for s in range(self.n_seeds):
            net = _Net(self.enc.n_num_, self.enc.cards_, counts.shape[1], self.hidden, self.dropout, emb_dims=self.enc.emb_dims_, tabm_k=self.tabm_k)
            self.nets_.append(_train(net, loss, tensors, val_mask, self.epochs, self.lr, self.batch, self.wd, self.patience, SEED + s, self.threads))
        return self

    def rate(self, X: NNX) -> np.ndarray:
        """(n, K) expected claims per unit of exposure, bounded at 20 x the base rate."""
        eta = np.mean([_predict(n, X) for n in self.nets_], axis=0)
        return np.minimum(np.exp(eta) * self.base_, 20.0 * self.base_)


class SevNet:
    """Gamma heads for K perils on claim-level (trended, capped) cost; rows = claims, peril_idx selects the head."""

    def __init__(self, enc: Encoder, hidden=(128, 64), dropout=0.1, epochs=20, lr=2e-3, batch=1024, wd=1e-3, patience=3, n_seeds=1, threads=4, tabm_k=0):
        self.enc, self.hidden, self.dropout, self.epochs, self.lr, self.batch, self.wd = enc, hidden, dropout, epochs, lr, batch, wd
        self.patience, self.n_seeds, self.threads, self.tabm_k = patience, n_seeds, threads, tabm_k

    def fit(self, X: NNX, peril_idx: np.ndarray, y: np.ndarray, k: int, val_mask: np.ndarray, w: np.ndarray | None = None):
        self.mean_ = np.array([y[peril_idx == j].mean() if (peril_idx == j).any() else y.mean() for j in range(k)])
        b0 = torch.tensor(np.log(self.mean_), dtype=torch.float32)
        tensors = [torch.from_numpy(X.num), torch.from_numpy(X.cat), torch.from_numpy(peril_idx.astype("int64")), torch.from_numpy(y.astype("float32")),
                   torch.from_numpy((np.ones(len(y)) if w is None else w).astype("float32"))]

        def loss(net, num, cat, pi, yy, ww):
            out = net(num, cat)
            out = out if out.dim() == 3 else out[:, None, :]
            eta = out.gather(2, pi[:, None, None].expand(-1, out.shape[1], 1))[:, :, 0] + b0[pi][:, None]
            per_row = (yy[:, None] * torch.exp(-eta) + eta).mean(1)
            return (per_row * ww).sum() / ww.sum()

        self.nets_ = []
        for s in range(self.n_seeds):
            net = _Net(self.enc.n_num_, self.enc.cards_, k, self.hidden, self.dropout, emb_dims=self.enc.emb_dims_, tabm_k=self.tabm_k)
            self.nets_.append(_train(net, loss, tensors, val_mask, self.epochs, self.lr, self.batch, self.wd, self.patience, SEED + s, self.threads))
        return self

    def mean(self, X: NNX) -> np.ndarray:
        """(n, K) expected severity per peril, bounded at 0.1 - 10 x the peril's training mean."""
        eta = np.mean([_predict(n, X) for n in self.nets_], axis=0)
        return np.clip(np.exp(eta) * self.mean_, 0.1 * self.mean_, 10.0 * self.mean_)
