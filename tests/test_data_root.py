# tests/test_data_root.py
"""DATA_ROOT lets the pipeline write outside the repo checkout (needed for CSC scratch)."""
import importlib
import os
import sys


def _reload_with_env(value):
    """Re-import config.const and src.training_pipeline with RAILWAY_DATA_ROOT set to `value`.

    Constants are bound at import time, so the modules must actually be reloaded.
    """
    if value is None:
        os.environ.pop("RAILWAY_DATA_ROOT", None)
    else:
        os.environ["RAILWAY_DATA_ROOT"] = value
    import config.const
    importlib.reload(config.const)
    import src.training_pipeline
    importlib.reload(src.training_pipeline)
    return config.const, src.training_pipeline


def test_data_root_defaults_to_empty_string():
    const, _ = _reload_with_env(None)
    assert const.DATA_ROOT == ""


def test_unset_data_root_resolves_to_repo_root(tmp_path):
    _, tp_module = _reload_with_env(None)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(tp_module.__file__)))
    pipeline = tp_module.TrainingPipeline()
    assert pipeline.project_root == repo_root


def test_set_data_root_relocates_output(tmp_path):
    target = str(tmp_path / "scratch")
    os.makedirs(target, exist_ok=True)
    _, tp_module = _reload_with_env(target)
    pipeline = tp_module.TrainingPipeline()
    assert pipeline.project_root == target
    assert pipeline.output_dir == os.path.join(target, "data/output")
    # __init__ creates the log dir eagerly; prove it landed under the new root
    assert os.path.isdir(os.path.join(target, "data", "output", "log"))
    _reload_with_env(None)  # restore for other tests


def test_ensure_folder_structure_honours_data_root(tmp_path):
    target = str(tmp_path / "scratch2")
    os.makedirs(target, exist_ok=True)
    os.environ["RAILWAY_DATA_ROOT"] = target
    import config.const
    importlib.reload(config.const)
    import src.file_utils
    importlib.reload(src.file_utils)
    src.file_utils.ensure_folder_structure()
    assert os.path.isdir(os.path.join(target, "data", "input"))
    assert os.path.isdir(os.path.join(target, "data", "output", "log"))
    os.environ.pop("RAILWAY_DATA_ROOT", None)
    importlib.reload(config.const)
    importlib.reload(src.file_utils)
