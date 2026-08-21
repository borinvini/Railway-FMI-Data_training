import numpy as np
import pytest

from src.fcll_trainer import compute_pos_weight, load_shards, train_fcll


def test_pos_weight_is_the_negative_to_positive_ratio():
    """Replaces SMOTE. Resampling would synthesise rows inside real journeys."""
    labels = np.array([0, 0, 0, 1])
    assert compute_pos_weight(labels) == pytest.approx(3.0)


def test_pos_weight_handles_an_all_negative_split():
    assert compute_pos_weight(np.zeros(10)) == pytest.approx(1.0)


def test_shards_are_loaded_and_concatenated(tmp_path):
    for i in range(2):
        np.savez_compressed(
            tmp_path / f"fcll_{i}.npz",
            z_seq=np.zeros((3, 5, 9), np.float32), h_seq=np.zeros((3, 5, 9), np.float32),
            z_mask=np.ones((3, 5), np.float32), h_mask=np.ones((3, 5), np.float32),
            nf=np.zeros((3, 2), np.float32), label=np.array([0, 1, 0], np.float32),
            group=np.array(["2018-01-01"] * 3),
        )
    data = load_shards(str(tmp_path))
    assert data["label"].shape == (6,)
    assert data["z_seq"].shape == (6, 5, 9)


def test_training_reduces_loss_on_a_learnable_fixture():
    """A model that cannot fit an obvious signal is mis-wired. The signal here
    is trivial: label == 1 exactly when the last station's delay is high."""
    rng = np.random.default_rng(42)
    n = 512
    z = np.zeros((n, 5, 9), np.float32)
    label = rng.integers(0, 2, n).astype(np.float32)
    z[:, -1, 0] = label * 10.0 + rng.normal(0, 0.1, n)
    data = {
        "z_seq": z, "h_seq": np.zeros((n, 5, 9), np.float32),
        "z_mask": np.ones((n, 5), np.float32), "h_mask": np.ones((n, 5), np.float32),
        "nf": np.zeros((n, 1), np.float32), "label": label,
        "group": np.array([f"d{i % 20}" for i in range(n)]),
    }
    result = train_fcll(data, epochs=15, batch_size=64, seed=42)
    assert result["history"][-1] < result["history"][0], "loss did not decrease"
    assert result["metrics"]["roc_auc"] > 0.9, "failed to learn a trivial signal"
