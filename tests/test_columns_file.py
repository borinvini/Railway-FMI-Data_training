"""Parsing of the plain-text feature file passed to `main.py --columns-file`.

The file is edited on Windows and copied to Linux with scp, so encoding and
line-ending handling are the substance of these tests, not incidental detail.
"""
import pytest

from config.columns_file import (
    has_scenarios,
    load_columns,
    load_scenarios,
    select_scenario,
)


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
    otherwise become part of every column name and fail validation.
    read_text() normalises the \\r\\n itself; .strip() is a backstop for any
    stray \\r plus ordinary surrounding whitespace, which is what the second
    case below actually exercises."""
    f = tmp_path / "features.txt"
    f.write_bytes(b"trainDelayed\r\nAir temperature (12h max)\r\n")
    assert load_columns(str(f)) == ["trainDelayed", "Air temperature (12h max)"]


def test_strips_surrounding_whitespace(tmp_path):
    """.strip()'s actual job: trimming stray leading/trailing spaces, which
    read_text()'s line-ending normalisation does not touch."""
    f = tmp_path / "features.txt"
    f.write_text("  Air temperature (12h max)  \n", encoding="utf-8")
    assert load_columns(str(f)) == ["Air temperature (12h max)"]


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
    assert "lines 1 and 3" in message


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


SCENARIO_FILE = """\
# Feature selection per training scenario.
# Blank lines and # comments are ignored.

# ======================================================================
# 1 - ALL FEATURES
# 2 columns
# ======================================================================
trainDelayed
Air temperature (12h max)

# ======================================================================
# 2 - ONLY OPERACIONAL FEATURES
# ======================================================================
trainDelayed
trainStopping
"""


def _scenario_file(tmp_path):
    f = tmp_path / "scenarios.txt"
    f.write_text(SCENARIO_FILE, encoding="utf-8")
    return str(f)


def test_load_scenarios_returns_sections_in_file_order(tmp_path):
    result = load_scenarios(_scenario_file(tmp_path))
    assert [name for name, _ in result] == [
        "1 - ALL FEATURES",
        "2 - ONLY OPERACIONAL FEATURES",
    ]
    assert result[0][1] == ["trainDelayed", "Air temperature (12h max)"]
    assert result[1][1] == ["trainDelayed", "trainStopping"]


def test_duplicate_detection_is_scoped_to_one_section(tmp_path):
    """Every scenario legitimately repeats trainDelayed, so the global check in
    load_columns must not apply across sections."""
    result = load_scenarios(_scenario_file(tmp_path))
    assert result[0][1][0] == "trainDelayed"
    assert result[1][1][0] == "trainDelayed"


def test_duplicate_within_one_section_is_rejected(tmp_path):
    f = tmp_path / "scenarios.txt"
    f.write_text("# 1 - A\ntrainDelayed\ntrainDelayed\n# 2 - B\ntrainStopping\n",
                 encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_scenarios(str(f))
    message = str(exc.value)
    assert "trainDelayed" in message
    assert "1 - A" in message


def test_non_banner_comments_are_not_treated_as_sections(tmp_path):
    """The `# ===` rules and the preamble must not open a section."""
    result = load_scenarios(_scenario_file(tmp_path))
    assert len(result) == 2


def test_empty_section_is_rejected(tmp_path):
    f = tmp_path / "scenarios.txt"
    f.write_text("# 1 - A\ntrainDelayed\n# 2 - EMPTY\n# 3 - C\ntrainStopping\n",
                 encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_scenarios(str(f))
    assert "2 - EMPTY" in str(exc.value)


def test_columns_before_the_first_banner_are_rejected(tmp_path):
    f = tmp_path / "scenarios.txt"
    f.write_text("trainDelayed\n# 1 - A\ntrainStopping\n# 2 - B\ncommercialStop\n",
                 encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_scenarios(str(f))
    assert "before any scenario banner" in str(exc.value)


def test_scenario_file_strips_bom_and_crlf(tmp_path):
    f = tmp_path / "scenarios.txt"
    f.write_bytes(b"\xef\xbb\xbf# 1 - A\r\ntrainDelayed\r\n# 2 - B\r\ntrainStopping\r\n")
    assert load_scenarios(str(f)) == [
        ("1 - A", ["trainDelayed"]),
        ("2 - B", ["trainStopping"]),
    ]


def test_select_scenario_by_index_is_one_based(tmp_path):
    name, columns = select_scenario(_scenario_file(tmp_path), 2)
    assert name == "2 - ONLY OPERACIONAL FEATURES"
    assert columns == ["trainDelayed", "trainStopping"]


def test_select_scenario_accepts_a_digit_string(tmp_path):
    """The batch script passes the index through argparse as a string."""
    name, _ = select_scenario(_scenario_file(tmp_path), "1")
    assert name == "1 - ALL FEATURES"


def test_select_scenario_by_exact_name(tmp_path):
    name, columns = select_scenario(_scenario_file(tmp_path), "1 - ALL FEATURES")
    assert name == "1 - ALL FEATURES"
    assert len(columns) == 2


def test_select_scenario_index_out_of_range_lists_the_catalogue(tmp_path):
    with pytest.raises(ValueError) as exc:
        select_scenario(_scenario_file(tmp_path), 9)
    message = str(exc.value)
    assert "1..2" in message
    assert "2 - ONLY OPERACIONAL FEATURES" in message, "the error must name the options"


def test_select_scenario_index_zero_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        select_scenario(_scenario_file(tmp_path), 0)


def test_select_scenario_unknown_name_lists_the_catalogue(tmp_path):
    with pytest.raises(ValueError) as exc:
        select_scenario(_scenario_file(tmp_path), "9 - NOT A SCENARIO")
    assert "1 - ALL FEATURES" in str(exc.value)


def test_has_scenarios_is_true_for_a_scenario_file(tmp_path):
    assert has_scenarios(_scenario_file(tmp_path)) is True


def test_has_scenarios_is_false_for_a_plain_feature_file(tmp_path):
    """A single stray comment like '# 12h block - see docs' must not make a
    plain features.txt look like a scenario file, which is why the threshold
    is two banners rather than one."""
    f = tmp_path / "features.txt"
    f.write_text("# 12h block - see docs\ntrainDelayed\ntrainStopping\n", encoding="utf-8")
    assert has_scenarios(str(f)) is False


def test_load_columns_on_a_scenario_file_says_to_pass_scenario(tmp_path):
    """Without this guard the repeated trainDelayed across sections would
    surface as a confusing duplicate-name error."""
    with pytest.raises(ValueError) as exc:
        load_columns(_scenario_file(tmp_path))
    message = str(exc.value)
    assert "2 scenario sections" in message
    assert "--scenario" in message
