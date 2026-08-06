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

CANDIDATES_RE = re.compile(r'CANDIDATES\+=\("([^"]+)"\)')


def _candidate_branches():
    """Split the script into the --train-all and array CANDIDATES blocks and
    extract the ordered list of globs each one appends.

    Relies on the script's shape: a single `if [ "${TRAIN_ALL}" -eq 1 ]; then
    ... else ... fi` block that builds CANDIDATES, ending right before the
    "Resolve first, transfer second" comment.
    """
    start = SCRIPT.index("CANDIDATES=()")
    end = SCRIPT.index("# Resolve first, transfer second")
    block = SCRIPT[start:end]
    train_all_part, array_part = block.split("\nelse\n", 1)
    return (
        CANDIDATES_RE.findall(train_all_part),
        CANDIDATES_RE.findall(array_part),
    )


def test_both_flags_are_handled_and_anything_else_is_rejected():
    assert "--models)" in SCRIPT
    assert "--train-all)" in SCRIPT
    assert "unknown argument" in SCRIPT


def test_candidate_globs_match_exactly_in_both_branches():
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
    train_all_actual, array_actual = _candidate_branches()
    assert train_all_actual == TRAIN_ALL_EXPECTED
    assert array_actual == ARRAY_EXPECTED


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
    assert SCRIPT.index("du -csh") < SCRIPT.index("scp -r")


def test_an_unreachable_cluster_points_at_the_auth_script():
    """The certificate lasts 24 h; expiry is the single most common failure and
    looks like a broken setup."""
    assert "hpc/roihu-auth.sh" in SCRIPT
