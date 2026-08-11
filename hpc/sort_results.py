"""Sort fetched run roots into per-scenario folders.

hpc/fetch-results.sh --scenarios brings back run_sNN_<model>/data/output/...
directories, because the cluster names everything by slug: scenario names
contain spaces, + and parentheses, and every remote path in that script goes
through an unquoted glob or `ls -d`. This module runs locally, after the
transfer, where those characters are safe.

A module rather than a heredoc so the test can import and call it — a copy of
the algorithm pasted into a test proves only that the copy works.
"""
import argparse
import shutil
import sys
from pathlib import Path

# hpc/ is not a package; add the repo root so `config` imports resolve when this
# is run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.columns_file import load_scenarios  # noqa: E402


def sort_results(results_dir, scenarios_file, log=print):
    """Move run_sNN_<model>/data/output/* into <results_dir>/<scenario name>/.

    One flat folder per scenario: the six 500-505 prep stages that scenario was
    built from, and the five 1000-1004 directories trained out of them.

    Flat works because of what the fetch selects, not by luck. The 100[0-4]-*
    directories are differently numbered per model, and hpc/fetch-results.sh
    takes 500-505 from each scenario's xgboost root ALONE — so exactly one root
    per scenario contributes prep stages and nothing collides. Widening that
    glob to run_s??_* without changing this function would silently let five
    identically-named copies overwrite each other.

    The slug -> name map comes from the same parser the cluster used, so the two
    cannot drift. A slug with no matching scenario is left in place rather than
    guessed at. Returns the number of run roots consumed.
    """
    results = Path(results_dir)
    names = {
        f"s{index:02d}": name
        for index, (name, _) in enumerate(load_scenarios(scenarios_file), start=1)
    }

    consumed = 0
    for root in sorted(results.glob("run_s??_*")):
        slug = root.name.split("_")[1]
        name = names.get(slug)
        if name is None:
            log(f"  ? {root.name}: slug {slug} is not in {scenarios_file} — left in place")
            continue
        output = root / "data" / "output"
        if not output.is_dir():
            log(f"  ? {root.name}: no data/output directory — left in place")
            continue

        target = results / name
        target.mkdir(parents=True, exist_ok=True)
        for stage in sorted(output.glob("*")):
            destination = target / stage.name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            shutil.move(str(stage), str(destination))
            log(f"  {destination.relative_to(results)}")
        shutil.rmtree(root)
        consumed += 1
    return consumed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Directory the fetch unpacked into.")
    parser.add_argument("scenarios_file", help="The scenario catalogue used for the run.")
    args = parser.parse_args(argv)
    sort_results(args.results_dir, args.scenarios_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
