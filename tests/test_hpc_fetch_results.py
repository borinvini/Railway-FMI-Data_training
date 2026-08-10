"""Static checks for hpc/fetch-results.sh.

The script calls ssh and scp, so it cannot run off the cluster; bash -n is the
only mechanical check available. What these tests protect is the part that is
easy to break with a careless edit — the globs, which decide whether the fetch
pulls 2 GB or 40 GB, and whether it silently drags the staged input back down.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (REPO_ROOT / "hpc" / "fetch-results.sh").read_text(encoding="utf-8")

SCRATCH = "${SCRATCH}"

# The exact, ordered candidate list each branch is expected to build. Update
# this alongside a deliberate change to hpc/fetch-results.sh's CANDIDATES
# logic — that is the point of pinning it here instead of scattering
# substring checks that a widened glob could slip past.
TRAIN_ALL_EXPECTED = [
    f"{SCRATCH}/data/output/50[0-5]-*",
    f"{SCRATCH}/data/output/700-*",
    f"{SCRATCH}/data/output/100[0-4]-*",
]
ARRAY_EXPECTED = [
    f"{SCRATCH}/run_0/data/output/50[0-5]-*",
    f"{SCRATCH}/run_0/data/output/700-*",
    f"{SCRATCH}/run_*/data/output/100[0-4]-*",
]
SCENARIOS_EXPECTED = [
    f"{SCRATCH}/run_s??_xgboost/data/output/50[0-5]-*",
    f"{SCRATCH}/run_s??_*/data/output/100[0-4]-*",
]

CANDIDATES_RE = re.compile(r'CANDIDATES\+=\("([^"]+)"\)')


def _candidate_branches():
    """Split the script into its three CANDIDATES blocks and extract the
    ordered list of globs each one appends.

    Relies on the script's shape: a single
    `if [ "${SCENARIOS}" -eq 1 ]; then ... elif [ "${TRAIN_ALL}" -eq 1 ]; then
    ... else ... fi` block, ending right before the "Resolve first, transfer
    second" comment. Splitting on the elif first is what keeps the scenarios
    globs from being swallowed into the train-all list.
    """
    start = SCRIPT.index("CANDIDATES=()")
    end = SCRIPT.index("# Resolve first, transfer second")
    block = SCRIPT[start:end]
    scenarios_part, rest = block.split('\nelif [ "${TRAIN_ALL}" -eq 1 ]; then\n', 1)
    train_all_part, array_part = rest.split("\nelse\n", 1)
    return (
        CANDIDATES_RE.findall(train_all_part),
        CANDIDATES_RE.findall(array_part),
        CANDIDATES_RE.findall(scenarios_part),
    )


def test_both_flags_are_handled_and_anything_else_is_rejected():
    assert "--models)" in SCRIPT
    assert "--train-all)" in SCRIPT
    assert "--scenarios)" in SCRIPT
    assert "--stages)" in SCRIPT
    assert "unknown argument" in SCRIPT


def test_candidate_globs_match_exactly_in_all_three_branches():
    """Structural regression test for the branch's core invariant: exactly
    which globs each layout fetches, in order, no more and no fewer.

    This exists because substring checks like `"run_0/.../50[0-5]-*" in
    SCRIPT` only prove the run_0 form is *present* — they say nothing about
    absence. A script that additionally (or instead) fetched
    `${SCRATCH}/run_*/data/output/50[0-5]-*` would still contain the run_0
    substring and pass such a check, while silently downloading five
    byte-identical copies of the merged/balanced/scaled prep-stage data per
    fetch — exactly the 5x-transfer waste this design exists to prevent.
    Comparing the full extracted CANDIDATES list against the exact expected
    set catches that widening, plus any other addition, removal, or glob
    change (e.g. an accidental `output/1*`), in one place.
    """
    train_all_actual, array_actual, scenarios_actual = _candidate_branches()
    assert train_all_actual == TRAIN_ALL_EXPECTED
    assert array_actual == ARRAY_EXPECTED
    assert scenarios_actual == SCENARIOS_EXPECTED


def test_the_absolute_path_arm_is_anchored():
    """The parser's catch-all arm for resolved output lines must require a
    leading `/`, not accept any non-empty line. A bare `*)` would treat a
    stray banner line ahead of the first `### ` marker (MOTD, a maintenance
    notice) as a directory to fetch and hand it straight to scp, aborting
    the whole transfer. Every real path from `ls -d` is absolute, since the
    globs themselves are absolute, so anchoring costs nothing."""
    assert "        /*)" in SCRIPT
    assert re.search(r'\n\s+\*\)\s*\n\s+PATHS\+=', SCRIPT) is None


def test_size_is_reported_before_the_transfer_starts():
    """The point of the du call is to learn a pull is 12 GB before it begins,
    not after."""
    assert "du -csh" in SCRIPT
    assert SCRIPT.index("du -csh") < SCRIPT.index("tar -czf -")


def test_the_transfer_opens_exactly_one_remote_connection():
    """scp opens a separate connection per remote source argument, so fetching
    11 directories authenticated 11 times. With no ssh-agent loaded that is 11
    passphrase prompts, one every couple of files — which is what this script
    shipped with and what this test exists to prevent coming back.

    A single `tar -czf -` stream carries every directory down one connection.
    Two ssh invocations total are expected: the resolve (which now also returns
    the sizes) and the transfer.
    """
    code = "\n".join(
        line for line in SCRIPT.splitlines() if not line.lstrip().startswith("#")
    )
    assert "scp" not in code, "scp costs one connection per source argument"
    assert "tar -czf -" in code
    assert code.count('ssh "${REMOTE_HOST}"') == 2


def test_the_archive_members_are_flattened_to_bare_directory_names():
    """The sources live under different run roots (run_0/... and run_*/...), so
    without a `-C <parent>` before each name the archive would carry the full
    /scratch/... prefix and results/ would gain a deep directory tree instead of
    the flat layout every doc and the compare-runs snippet assume."""
    assert '-C $(dirname "${path}") $(basename "${path}")' in SCRIPT
    assert "-C results" in SCRIPT


def test_an_unreachable_cluster_points_at_the_auth_script():
    """The certificate lasts 24 h; expiry is the single most common failure and
    looks like a broken setup."""
    assert "hpc/roihu-auth.sh" in SCRIPT


def test_the_scenarios_flag_is_handled():
    assert "--scenarios)" in SCRIPT
    assert "--stages)" in SCRIPT


def test_scenarios_mode_puts_the_prep_stages_behind_a_flag():
    """Inverted relative to the other modes: with 8 distinct prep sets instead
    of one, pulling them every time is the wrong default.

    Asserted by position against the inner `fi`, not by slicing to the end of
    the branch: the model glob sits immediately after that `fi`, so an
    unbounded slice would always contain it and the test could never fail.
    """
    block = SCRIPT[SCRIPT.index("CANDIDATES=()"):SCRIPT.index("# Resolve first, transfer second")]
    scenarios_part = block.split('\nelif [ "${TRAIN_ALL}" -eq 1 ]; then\n', 1)[0]

    stages_open = scenarios_part.index('if [ "${STAGES}" -eq 1 ]; then')
    stages_close = scenarios_part.index("\n    fi", stages_open)
    prep_at = scenarios_part.index("50[0-5]-*")
    models_at = scenarios_part.index("100[0-4]-*")

    assert stages_open < prep_at < stages_close, "prep stages must be inside the --stages guard"
    assert models_at > stages_close, "model dirs must be outside it"


def test_scenarios_prep_comes_from_one_run_root_per_scenario():
    """run_s??_xgboost, not run_s??_*: the five models within a scenario produce
    byte-identical prep output, so the other four are pure transfer waste."""
    assert f"{SCRATCH}/run_s??_xgboost/data/output/50[0-5]-*" in SCRIPT
    assert f"{SCRATCH}/run_s??_*/data/output/50[0-5]-*" not in SCRIPT


def test_scenarios_mode_archives_members_relative_to_scratch():
    """Flattening to basenames would collapse all 8 scenarios' 1000-* dirs onto
    each other. The run-root prefix is what carries the scenario home."""
    assert "-C ${SCRATCH} ${path#${SCRATCH}/}" in SCRIPT


def test_the_reshuffle_calls_the_shared_module_not_an_inline_copy():
    """The sorting logic lives in hpc/sort_results.py so the test suite can
    exercise the shipped code; an inline heredoc could only be tested by a copy."""
    assert "python hpc/sort_results.py results config/scenarios.txt" in SCRIPT
    assert "load_scenarios" not in SCRIPT, "the parser is the module's job, not the script's"


def test_the_reshuffle_degrades_gracefully_without_python():
    """Git Bash on Windows may have no python on PATH; the transfer must still
    complete, leaving the slugged directories in place."""
    assert "command -v python" in SCRIPT


def test_scenarios_mode_still_opens_exactly_one_transfer_connection():
    code = "\n".join(
        line for line in SCRIPT.splitlines() if not line.lstrip().startswith("#")
    )
    assert code.count('ssh "${REMOTE_HOST}"') == 2
