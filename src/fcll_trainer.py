"""Training loop for FCLL-Net.

Kept out of training_pipeline.py, which is already ~3000 lines, and separate
from fcll_model.py so the architecture can be unit-tested without touching an
optimiser.
"""
import glob
import os

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

from config.const_training import (FCLL_VAL_FRACTION, HOLDOUT_LAST_YEAR,
                                   RANDOM_STATE)
from src.day_split import split_days
from src.fcll_model import FCLLNet

KEYS = ("z_seq", "h_seq", "z_mask", "h_mask", "nf", "label")


def compute_pos_weight(labels):
    """Negative-to-positive ratio for BCEWithLogitsLoss.

    Returns 1.0 for a degenerate split rather than dividing by zero: a split
    with no positives is a bug for the caller to catch, not something to make
    infinite here.
    """
    pos = float(np.sum(labels))
    neg = float(len(labels) - pos)
    return neg / pos if pos > 0 else 1.0


def load_shards(data_dir):
    """Concatenate every .npz shard in `data_dir`."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"no .npz shards in {data_dir}")
    parts = [np.load(p, allow_pickle=True) for p in paths]
    out = {k: np.concatenate([p[k] for p in parts]) for k in KEYS}
    out["group"] = np.concatenate([p["group"] for p in parts])
    return out


def _holdout_mask(groups):
    """Rows of the newest year, like the tabular pipeline's out-of-time hold-out.

    Returns (mask, year). Groups are operating days ("2025-03-14"); when they do not
    start with a year there is no hold-out and the mask is all False.
    """
    none = np.zeros(len(groups), dtype=bool)
    if not HOLDOUT_LAST_YEAR:
        return none, None
    try:
        years = np.array([int(str(g)[:4]) for g in groups])
    except ValueError:
        return none, None
    year = int(years.max())
    return years == year, year


def _split_indices(groups, holdout, seed):
    """Masks for train/val/test/holdout.

    Holdout rows (newest year) are never trained, validated or tested on. The rest is
    split by day 80/20 with the same rule as the tabular models (src/day_split), and
    the early-stopping days are cut from the TRAIN days only, so the test days stay
    untouched by training and by the choice of the stopping epoch.
    """
    train_days, test_days = split_days(groups[~holdout], seed=seed)
    ordered = np.array(sorted(train_days))
    np.random.default_rng(seed).shuffle(ordered)
    n_val = int(round(len(ordered) * FCLL_VAL_FRACTION))
    sets = {"val": set(ordered[:n_val]), "train": set(ordered[n_val:]), "test": test_days}
    masks = {k: np.isin(groups, list(v)) & ~holdout for k, v in sets.items()}
    masks["holdout"] = holdout
    return masks


def train_fcll(data, epochs=150, batch_size=2048, lr=1e-3, seed=RANDOM_STATE,
               patience=10):
    """Train on a day-grouped split of `data`; return metrics on the test split and,
    when the data spans several years, on the never-seen newest year (`holdout`)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    holdout, holdout_year = _holdout_mask(data["group"])
    idx = _split_indices(data["group"], holdout, seed)
    for name in ("train", "val", "test"):
        if not idx[name].any():
            raise ValueError(f"the {name} split is empty — too few operating days")

    def loader(split, shuffle):
        tensors = [torch.from_numpy(np.ascontiguousarray(data[k][idx[split]])) for k in KEYS]
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)

    model = FCLLNet(n_of=data["z_seq"].shape[2], n_nf=data["nf"].shape[1])
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(compute_pos_weight(data["label"][idx["train"]]))
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    history, best, since_best, best_state = [], float("inf"), 0, None
    for _ in range(epochs):
        model.train()
        for *inputs, y in loader("train", True):
            opt.zero_grad()
            loss = criterion(model(*inputs), y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            losses = [criterion(model(*i), y).item() for *i, y in loader("val", False)]
        val_loss = float(np.mean(losses)) if losses else float("inf")
        history.append(val_loss)
        sched.step(val_loss)

        if val_loss < best:
            best, since_best = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()

    def score(split):
        probs, trues = [], []
        with torch.no_grad():
            for *inputs, y in loader(split, False):
                probs.append(torch.sigmoid(model(*inputs)).numpy())
                trues.append(y.numpy())
        prob = np.concatenate(probs)
        true = np.concatenate(trues)
        pred = (prob >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(true, pred)),
            "precision": float(precision_score(true, pred, zero_division=0)),
            "recall": float(recall_score(true, pred, zero_division=0)),
            "f1": float(f1_score(true, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(true, prob)) if len(set(true)) > 1 else float("nan"),
            "confusion_matrix": confusion_matrix(true, pred).tolist(),
            "n": int(len(true)),
        }

    test = score("test")
    metrics = {
        **{k: v for k, v in test.items() if k != "n"},
        "n_train": int(idx["train"].sum()),
        "n_val": int(idx["val"].sum()),
        "n_test": test["n"],
        "n_holdout": int(idx["holdout"].sum()),
        "holdout_year": holdout_year,
        "epochs_run": len(history),
        "best_val_loss": float(best),
    }
    if idx["holdout"].any():
        metrics["holdout"] = score("holdout")

    return {"model": model, "history": history, "metrics": metrics}
