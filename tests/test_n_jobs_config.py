"""Parallelism constants must respect the Slurm allocation, not the whole node."""
import importlib
import inspect
import os


def _reload_const_training(slurm_cpus, monkeypatch, railway_n_jobs=None):
    if slurm_cpus is None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    else:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(slurm_cpus))
    if railway_n_jobs is None:
        monkeypatch.delenv("RAILWAY_N_JOBS", raising=False)
    else:
        monkeypatch.setenv("RAILWAY_N_JOBS", str(railway_n_jobs))
    import config.const_training
    importlib.reload(config.const_training)
    return config.const_training


def test_search_n_jobs_defaults_to_minus_one_off_cluster(monkeypatch):
    ct = _reload_const_training(None, monkeypatch)
    try:
        assert ct.SEARCH_N_JOBS == -1
    finally:
        importlib.reload(ct)


def test_search_n_jobs_reads_slurm_cpus_per_task(monkeypatch):
    ct = _reload_const_training(40, monkeypatch)
    try:
        assert ct.SEARCH_N_JOBS == 40
    finally:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        importlib.reload(ct)


def test_search_n_jobs_prefers_railway_n_jobs_over_slurm(monkeypatch):
    """--n-jobs (RAILWAY_N_JOBS) must win over Slurm's own allocation record,
    since it does not overwrite SLURM_CPUS_PER_TASK anymore (see main.py)."""
    ct = _reload_const_training(40, monkeypatch, railway_n_jobs=7)
    try:
        assert ct.SEARCH_N_JOBS == 7
    finally:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.delenv("RAILWAY_N_JOBS", raising=False)
        importlib.reload(ct)


def test_search_n_jobs_falls_back_to_slurm_without_railway_n_jobs(monkeypatch):
    """With no explicit --n-jobs, the allocation-aware SLURM_CPUS_PER_TASK path
    must still work on its own, since the batch scripts rely on it directly."""
    ct = _reload_const_training(40, monkeypatch, railway_n_jobs=None)
    try:
        assert ct.SEARCH_N_JOBS == 40
    finally:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        importlib.reload(ct)


def test_model_n_jobs_is_one(monkeypatch):
    """Inner estimators must be single-threaded so outer CV workers get the cores."""
    ct = _reload_const_training(None, monkeypatch)
    try:
        assert ct.MODEL_N_JOBS == 1
    finally:
        importlib.reload(ct)


def test_search_iterations_env_override(monkeypatch):
    monkeypatch.setenv("RAILWAY_SEARCH_ITERATIONS", "10")
    import config.const_training
    try:
        importlib.reload(config.const_training)
        assert config.const_training.RANDOM_SEARCH_ITERATIONS == 10
    finally:
        monkeypatch.delenv("RAILWAY_SEARCH_ITERATIONS", raising=False)
        importlib.reload(config.const_training)
    assert config.const_training.RANDOM_SEARCH_ITERATIONS == 50


def test_no_hardcoded_n_jobs_literals_remain():
    """Guards against a regression to whole-node oversubscription."""
    import src.training_pipeline as tp
    source = inspect.getsource(tp)
    assert "n_jobs=-1" not in source, "hardcoded n_jobs=-1 still present"
