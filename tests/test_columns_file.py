"""Parsing of the plain-text feature file passed to `main.py --columns-file`.

The file is edited on Windows and copied to Linux with scp, so encoding and
line-ending handling are the substance of these tests, not incidental detail.
"""
import pytest

from config.columns_file import load_columns


def test_reads_one_name_per_line(tmp_path):
    f = tmp_path / "features.txt"
    f.write_text("trainDelayed\ntrainStopping\n", encoding="utf-8")
    assert load_columns(str(f)) == ["trainDelayed", "trainStopping"]


def test_preserves_spaces_and_parentheses_verbatim(tmp_path):
    """The real column names look like this; they must survive unquoted."""
    f = tmp_path / "features.txt"
    f.write_text("Air temperature (12h max)\nPrecipitation amount (72h cumulative)\n",
                 encoding="utf-8")
    assert load_columns(str(f)) == [
        "Air temperature (12h max)",
        "Precipitation amount (72h cumulative)",
    ]


def test_strips_crlf_line_endings(tmp_path):
    """A CRLF file scp'd from Windows to Linux keeps its \\r, which would
    otherwise become part of every column name and fail validation."""
    f = tmp_path / "features.txt"
    f.write_bytes(b"trainDelayed\r\nAir temperature (12h max)\r\n")
    assert load_columns(str(f)) == ["trainDelayed", "Air temperature (12h max)"]


def test_strips_utf8_bom(tmp_path):
    """Windows editors prepend a BOM, which would corrupt the first name."""
    f = tmp_path / "features.txt"
    f.write_bytes(b"\xef\xbb\xbftrainDelayed\n")
    assert load_columns(str(f)) == ["trainDelayed"]


def test_ignores_comments_and_blank_lines(tmp_path):
    f = tmp_path / "features.txt"
    f.write_text(
        "# temperature block\n"
        "\n"
        "trainDelayed\n"
        "   \n"
        "  # indented comment\n"
        "trainStopping\n",
        encoding="utf-8",
    )
    assert load_columns(str(f)) == ["trainDelayed", "trainStopping"]


def test_duplicate_name_is_rejected_with_both_line_numbers(tmp_path):
    """training_pipeline.py:1356 would otherwise select the column twice."""
    f = tmp_path / "features.txt"
    f.write_text("trainDelayed\ntrainStopping\ntrainDelayed\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_columns(str(f))
    message = str(exc.value)
    assert "trainDelayed" in message
    assert "1" in message and "3" in message


def test_empty_file_is_rejected(tmp_path):
    """An empty selection drops select_training_cols into its interactive
    prompt (training_pipeline.py:1360), which hangs for the whole walltime."""
    f = tmp_path / "features.txt"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_columns(str(f))


def test_comments_only_file_is_rejected(tmp_path):
    f = tmp_path / "features.txt"
    f.write_text("# everything commented out\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_columns(str(f))


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_columns(str(tmp_path / "nope.txt"))


def test_env_var_overrides_the_frozen_selected_columns(tmp_path, monkeypatch, capsys):
    """RAILWAY_COLUMNS_FILE replaces the literal list, and the resolved list is
    printed so a finished slurm log identifies the feature set that ran."""
    import importlib
    import config.const_training as const_training

    features = tmp_path / "features.txt"
    features.write_text("trainDelayed\nAir temperature (12h max)\n", encoding="utf-8")
    monkeypatch.setenv("RAILWAY_COLUMNS_FILE", str(features))

    try:
        importlib.reload(const_training)
        assert const_training.SELECTED_COLUMNS == [
            "trainDelayed",
            "Air temperature (12h max)",
        ]
        out = capsys.readouterr().out
        assert str(features) in out
        assert "Air temperature (12h max)" in out, "the full list must reach the log"
    finally:
        # Restore the module for every other test in the session: monkeypatch
        # would undo the variable only after this test's reload had baked it in.
        monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)
        importlib.reload(const_training)


def test_without_the_env_var_the_frozen_list_is_used(monkeypatch):
    import importlib
    import config.const_training as const_training

    monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)
    importlib.reload(const_training)
    assert len(const_training.SELECTED_COLUMNS) == 79
    assert "trainDelayed" in const_training.SELECTED_COLUMNS
