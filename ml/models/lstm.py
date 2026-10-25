"""Small LSTM for both tasks (Phase 5 Part B). Plugs into the Part A runner unchanged.

Input per sample
    sequence : 24 hourly steps ending at the prediction time. For each channel (vital or lab) three signals:
               value (scaled), "newly measured in this hour" flag, hours since last measurement (scaled)
    static   : the context features of the task (age, sex, care unit, ...), scaled
Everything that is learned from data (median fill, scaling, early stopping) uses ONLY the training rows that
fit_predict() is given, so the test fold cannot influence it.

Fixed settings, no tuning (see docs/scope.md): 1 LSTM layer with 32 units, dropout, weight decay, class weighting,
early stopping on a patient-grouped slice of the training rows, 3 seeds averaged.
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from ml.config import DATA_PROCESSED
from ml.features.windows import feature_groups
from ml.models.feature_sets import design_matrix, load_task

CHANNEL_SETS = {"all": slice(0, 19), "vitals": slice(0, 7)}          # 7 vitals first, then 12 core labs
SETTINGS = dict(hidden=32, dropout=0.3, lr=3e-3, weight_decay=1e-2, max_epochs=30, patience=5, n_seeds=3,
                val_share=5)                                           # val_share=5 -> one fifth of the training patients


# --------------------------------------------------------------------------- data
def load_sequence_data(task: str, channels: str = "all"):
    """Returns dict(X, M, D, static, y, groups). Rows are in the same order as the tabular table (checked)."""
    z = np.load(DATA_PROCESSED / f"{task}_sequences.npz", allow_pickle=True)
    df, y, groups = load_task(task)
    assert (z["stay_id"] == df["stay_id"].to_numpy()).all() and (z["y"] == y).all(), "sequence rows do not match features"
    ch = CHANNEL_SETS[channels]
    static = design_matrix(df, feature_groups(df.columns)["context"]).to_numpy()
    return dict(X=z["X"][:, :, ch].astype(np.float32), M=z["M"][:, :, ch].astype(np.float32),
                D=z["D"][:, :, ch].astype(np.float32), static=static.astype(np.float32), y=y, groups=groups)


# --------------------------------------------------------------------------- preprocessing (fitted on training rows only)
class Prep:
    """Median fill + standardisation learned on the training rows."""

    def __init__(self, X, static):
        flat = X.reshape(-1, X.shape[2])
        self.med = np.nanmedian(flat, axis=0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)             # channel never observed in training
        filled = np.where(np.isnan(flat), self.med, flat)
        self.mu, self.sd = filled.mean(0), filled.std(0) + 1e-6
        self.s_med = np.nanmedian(static, axis=0)
        self.s_med = np.where(np.isnan(self.s_med), 0.0, self.s_med)
        s_filled = np.where(np.isnan(static), self.s_med, static)
        self.s_mu, self.s_sd = s_filled.mean(0), s_filled.std(0) + 1e-6

    def sequence(self, X, M, D):
        x = (np.where(np.isnan(X), self.med, X) - self.mu) / self.sd
        d = np.log1p(D) / np.log1p(48.0)                                      # hours since last measurement, 0..1
        return np.concatenate([x, M, d], axis=2).astype(np.float32)

    def static_(self, S):
        return ((np.where(np.isnan(S), self.s_med, S) - self.s_mu) / self.s_sd).astype(np.float32)


# --------------------------------------------------------------------------- network
class Net(nn.Module):
    def __init__(self, n_in, n_static, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([self.drop(h[-1]), static], dim=1)).squeeze(1)


def _train_one(seq, st, y, val_mask, seed, cfg, pos_weight):
    torch.manual_seed(seed)
    net = Net(seq.shape[2], st.shape[1], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
    tr, va = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    S, T, Y = torch.from_numpy(seq), torch.from_numpy(st), torch.from_numpy(y.astype(np.float32))
    bs = 128 if len(tr) > 500 else 16
    rng = np.random.default_rng(seed)
    best, best_state, wait = np.inf, None, 0
    for _ in range(cfg["max_epochs"]):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            opt.zero_grad()
            lossf(net(S[b], T[b]), Y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(S[va], T[va]), Y[va]))
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def lstm_fit_predict(data: dict, **overrides):
    """Returns fit_predict(train_idx, eval_idx, seed) for the CV runner."""
    cfg = {**SETTINGS, **overrides}
    X, M, D, static, y, groups = (data[k] for k in ["X", "M", "D", "static", "y", "groups"])

    def fit_predict(train_idx, eval_idx, seed):
        train_idx, eval_idx = np.asarray(train_idx), np.asarray(eval_idx)
        ytr = y[train_idx]
        if ytr.sum() == 0:
            return np.zeros(len(eval_idx))
        torch.set_num_threads(1)
        prep = Prep(X[train_idx], static[train_idx])                          # fitted on training rows only
        seq_tr, st_tr = prep.sequence(X[train_idx], M[train_idx], D[train_idx]), prep.static_(static[train_idx])
        seq_ev, st_ev = prep.sequence(X[eval_idx], M[eval_idx], D[eval_idx]), prep.static_(static[eval_idx])
        pos_weight = (len(ytr) - ytr.sum()) / ytr.sum()
        probs = []
        for k in range(cfg["n_seeds"]):
            s = seed * 10 + k
            # early-stopping slice: a patient-grouped fifth of the TRAINING rows (never the test fold)
            cv = StratifiedGroupKFold(n_splits=cfg["val_share"], shuffle=True, random_state=s)
            _, val = next(cv.split(np.zeros(len(ytr)), ytr, groups[train_idx]))
            val_mask = np.zeros(len(ytr), dtype=bool)
            val_mask[val] = True
            net = _train_one(seq_tr, st_tr, ytr, val_mask, s, cfg, pos_weight)
            with torch.no_grad():
                probs.append(torch.sigmoid(net(torch.from_numpy(seq_ev), torch.from_numpy(st_ev))).numpy())
        return np.mean(probs, axis=0)

    return fit_predict
