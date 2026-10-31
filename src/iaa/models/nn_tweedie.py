"""Direct Tweedie neural net on the row cost (one network per line): the structural alternative to the frequency x severity nets, for blending.

mu = exposure x exp(net(x)) with embeddings + a linear skip layer (models/nn.py), trained on the Tweedie(p) deviance of the row cost.
Row costs are winsorised at a high quantile of the positive costs for training only (one giant claim would dominate a deviance gradient);
predictions are then rescaled to the training total of the unwinsorised cost, as the direct LightGBM baseline does (Tweedie with a log link does not
preserve totals). No trend / completeness adjustment: the net learns the as-valued cost level of its training years.
"""
import numpy as np
import pandas as pd
import torch

from ..config import SEED, TWEEDIE_POWER
from ..features import feature_columns
from . import nn as nnm


class NNTweedie:
    name = "nn_tweedie"

    def __init__(self, cap_quantile: float = 0.9995, drop_noise: bool = True, extra_drop=(), hidden=(192, 96), dropout=0.1, epochs=12, lr=2e-3,
                 batch=2048, wd=1e-4, patience=2, n_seeds=1, threads=4, rebalance: bool = True):
        self.cap_quantile, self.drop_noise, self.extra_drop = cap_quantile, drop_noise, tuple(extra_drop)
        self.hidden, self.dropout, self.epochs, self.lr, self.batch, self.wd = hidden, dropout, epochs, lr, batch, wd
        self.patience, self.n_seeds, self.threads, self.rebalance = patience, n_seeds, threads, rebalance

    def fit(self, train: pd.DataFrame):
        cols = feature_columns(train, self.drop_noise, self.extra_drop)
        num = [c for c in cols if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
        self.enc_ = nnm.Encoder(num, [c for c in cols if c not in num]).fit(train)
        X = self.enc_.transform(train)
        y = train["y"].to_numpy(dtype=float)
        pos = y[y > 0]
        self.cap_ = float(np.quantile(pos, self.cap_quantile))
        yc = np.minimum(y, self.cap_).astype("float32")
        expo = train["exposure_calc"].to_numpy(dtype="float32")
        self.base_ = float(yc.sum() / expo.sum())
        b0 = float(np.log(self.base_))
        val = pd.util.hash_array(train["customer_id"].astype(str).to_numpy()) % 20 == 0
        p = TWEEDIE_POWER
        tensors = [torch.from_numpy(X.num), torch.from_numpy(X.cat), torch.from_numpy(yc), torch.from_numpy(np.log(np.maximum(expo, 1e-6)))]

        def loss(net, num_, cat, yy, logoff):
            eta = net(num_, cat)[:, 0] + b0 + logoff
            return (-yy * torch.exp((1 - p) * eta) / (1 - p) + torch.exp((2 - p) * eta) / (2 - p)).mean()

        self.nets_ = []
        for s in range(self.n_seeds):
            net = nnm._Net(self.enc_.n_num_, self.enc_.cards_, 1, self.hidden, self.dropout)
            self.nets_.append(nnm._train(net, loss, tensors, val, self.epochs, self.lr, self.batch, self.wd, self.patience, SEED + s, self.threads))
        self.k_ = 1.0
        if self.rebalance:
            self.k_ = float(y.sum() / self._raw(train).sum())
        return self

    def _raw(self, rows: pd.DataFrame) -> np.ndarray:
        eta = np.mean([nnm._predict(n, self.enc_.transform(rows))[:, 0] for n in self.nets_], axis=0)
        return np.exp(eta + np.log(self.base_)) * rows["exposure_calc"].to_numpy(dtype=float)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.k_ * self._raw(rows)
