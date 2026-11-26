"""Stage and fetch the multi-station grid from Windows without Git Bash.

hpc/stage_data.sh and hpc/fetch-stations.sh are bash. On a machine where Git Bash
cannot run (the university machine: "Access is denied") this script does the
same with Windows' own OpenSSH (C:\\Windows\\System32\\OpenSSH\\ssh.exe) and
Python, and it uses ONE ssh connection per command, because without an ssh agent
every connection asks for the key passphrase.

    python -X utf8 scripts/roihu_windows.py stage [--stations all|OL,ROI]
    python -X utf8 scripts/roihu_windows.py fetch [--stations ...] [--archive DIR] [--no-archive]

stage: data/stations/<ST>/data/output/101-preprocessed_training_ready -> Roihu
       /scratch/project_2019266/railway-fmi/stations/<ST>/data/output/..., as one
       tar stream. Each station's old 101 folder on Roihu is replaced, so no stale
       month survives. Ends with the file count and content hash per station on
       both sides (the md5 of the per-file md5 column, like stage_data.sh).
fetch: results/<ST>/run_sNN_<model>/... from every station in one tar stream, then
       hpc/sort_results.py (retried on Windows/OneDrive locks), hpc/grid_summary.py,
       and a copy into <archive>/<N> - <ST>/results (refused up front if not empty).
Needs the `roihu` alias in ~/.ssh/config and today's certificate (docs/CSC-SETUP.md).
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.dirname(SCRIPTS))

from stations import REPO_ROOT, archive_folder_name, load_stations, select_stations  # noqa: E402

HOST = "roihu"
SCRATCH = "/scratch/project_2019266/railway-fmi"
PREPROCESSED = "data/output/101-preprocessed_training_ready"
EXPECTED_FILES = 96
DEFAULT_ARCHIVE = os.path.join(
    os.path.dirname(REPO_ROOT), "Railway-FMI-Data-CSV-Files-v2", "AI Results", "NUMERIC")
SSH = shutil.which("ssh") or r"C:\Windows\System32\OpenSSH\ssh.exe"
SCP = shutil.which("scp") or r"C:\Windows\System32\OpenSSH\scp.exe"
TAR = shutil.which("tar") or r"C:\Windows\System32\tar.exe"


def folder_hash(folder):
    """(count, hash) of a folder's *.parquet, equal to the remote
    `md5sum *.parquet | cut -c1-32 | md5sum | cut -c1-32`: the md5 of the lines
    "<md5 of file>\\n", files in name order."""
    names = sorted(n for n in os.listdir(folder) if n.endswith(".parquet"))
    lines = []
    for name in names:
        digest = hashlib.md5()
        with open(os.path.join(folder, name), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        lines.append(digest.hexdigest() + "\n")
    return len(names), hashlib.md5("".join(lines).encode()).hexdigest()


UPLOAD = f"{SCRATCH}/stations_upload.tar"


def stage_remote_command(stations):
    """The shell run on Roihu: replace each station's 101 folder from the uploaded tar, then report.

    The tar arrives by scp first, not on ssh's stdin: with stdin a pipe, Windows
    OpenSSH closed the connection right after the passphrase prompt (first run).
    """
    folders = " ".join(f"{st}/{PREPROCESSED}" for st in stations)
    report = (f"for st in {' '.join(stations)}; do cd {SCRATCH}/stations/$st/{PREPROCESSED} && "
              "echo STAGED $st $(ls -1 *.parquet | wc -l) "
              "$(md5sum *.parquet | cut -c1-32 | md5sum | cut -c1-32); done")
    # chmod u+w: Windows tar archives the folders read-only, and a read-only folder
    # on Roihu makes its files undeletable ("Permission denied" on the first cleanup).
    owned = " ".join(f"{st}" for st in stations)
    return (f"set -e; mkdir -p {SCRATCH}/stations && cd {SCRATCH}/stations && "
            f"(chmod -R u+w {owned} 2>/dev/null || true) && rm -rf {folders} "
            f"&& tar -xf {UPLOAD} && chmod -R u+w {owned} && rm -f {UPLOAD} && {report}")


def stage(args):
    stations = select_stations(args.stations)
    local = {}
    for st in stations:
        folder = os.path.join(REPO_ROOT, "data", "stations", st, *PREPROCESSED.split("/"))
        if not os.path.isdir(folder):
            sys.exit(f"ERROR: {folder} not found. Run scripts/preprocess_stations.py first.")
        local[st] = folder_hash(folder)
        if local[st][0] != EXPECTED_FILES:
            sys.exit(f"ERROR: {st} has {local[st][0]} files locally, expected {EXPECTED_FILES}.")
    members = [f"{st}/{PREPROCESSED}" for st in stations]
    archive = os.path.join(tempfile.gettempdir(), "railway_stations_upload.tar")
    print(f"Packing {len(stations)} stations: {' '.join(stations)} -> {archive}", flush=True)
    tar = subprocess.run([TAR, "-cf", archive, "-C", os.path.join(REPO_ROOT, "data", "stations"),
                          *members])
    if tar.returncode:
        print(f"ERROR: tar exited {tar.returncode}; nothing was sent.")
        return 1
    print(f"Uploading {os.path.getsize(archive) / 1e6:.0f} MB (passphrase 1 of 2)...", flush=True)
    scp = subprocess.run([SCP, archive, f"{HOST}:{UPLOAD}"])
    if scp.returncode:
        print(f"ERROR: scp exited {scp.returncode}; nothing on Roihu was changed. Run it again.")
        return 1
    print("Unpacking on Roihu and hashing (passphrase 2 of 2)...", flush=True)
    ssh = subprocess.run([SSH, HOST, stage_remote_command(stations)],
                         stdout=subprocess.PIPE, text=True)
    os.remove(archive)
    remote = {}
    for line in ssh.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "STAGED":
            remote[parts[1]] = (int(parts[2]), parts[3])
    bad = 0
    print()
    for st in stations:
        n, h = local[st]
        rn, rh = remote.get(st, (None, None))
        verdict = "MATCH" if (n, h) == (rn, rh) else "DIFFER"
        bad += verdict != "MATCH"
        print(f"{st}: local {n} files {h} | Roihu {rn} files {rh} | {verdict}")
    if ssh.returncode or bad:
        print(f"ERROR: staging incomplete (ssh exit {ssh.returncode}, {bad} stations differ). "
              "Run it again.")
        return 1
    print(f"All {len(stations)} stations staged under {HOST}:{SCRATCH}/stations/")
    return 0


DOWNLOAD = f"{SCRATCH}/stations_results.tar.gz"


def fetch_remote_command(stations):
    """Pack each station's prep (from the xgboost roots) and model folders into DOWNLOAD.

    Written to a file on Roihu and copied down by scp, never streamed on ssh's
    stdout: an ssh started by this script with a captured stream closed the
    connection right after the passphrase prompt (staging, twice), while scp and
    an ssh typed in the console worked.
    """
    globs = []
    for st in stations:
        globs += [f"{st}/run_s??_xgboost/data/output/50[0-5]-*",
                  f"{st}/run_s??_*/data/output/100[0-4]-*",
                  f"{st}/run_s??_*/data/output/999-*"]
    return (f"cd {SCRATCH}/stations && paths=$(ls -d {' '.join(globs)} 2>/dev/null); "
            "[ -n \"$paths\" ] || { echo NOTHING-TO-FETCH; exit 1; }; "
            f"tar -czf {DOWNLOAD} $paths && ls -l {DOWNLOAD}")


def _sort_with_retries(results_dir, scenarios_file, attempts=60):
    from hpc.sort_results import sort_results
    for _ in range(attempts):
        try:
            sort_results(results_dir, scenarios_file, log=lambda *_: None)
        except PermissionError:
            time.sleep(1)
        left = [n for n in os.listdir(results_dir) if n.startswith("run_s")]
        for name in left:  # empty shells left by a locked move
            path = os.path.join(results_dir, name)
            for root, dirs, files in os.walk(path, topdown=False):
                if not files and not os.listdir(root):
                    os.rmdir(root)
        if not [n for n in os.listdir(results_dir) if n.startswith("run_s")]:
            return 0
    return len([n for n in os.listdir(results_dir) if n.startswith("run_s")])


def _count(results_dir):
    scen = [d for d in os.listdir(results_dir) if d[:1].isdigit() and " - " in d]
    n_json = 0
    for d in scen:
        for sub in os.listdir(os.path.join(results_dir, d)):
            if sub[:4] in ("1000", "1001", "1002", "1003", "1004") or sub.startswith("999-"):
                n_json += sum(1 for f in os.listdir(os.path.join(results_dir, d, sub))
                              if f.endswith(".json") and ("iteration_analysis" in f
                                                          or f.startswith("ridge_baseline_")))
    return len(scen), n_json


def fetch(args):
    from hpc import grid_summary
    stations = select_stations(args.stations)
    all_stations = load_stations()
    if not args.no_archive:
        if not os.path.isdir(args.archive):
            sys.exit(f"ERROR: archive folder not found: {args.archive}")
        for st in stations:
            target = os.path.join(args.archive, archive_folder_name(st, all_stations), "results")
            if os.path.isdir(target) and os.listdir(target):
                sys.exit(f"ERROR: {target} is not empty. Move it aside first; nothing was fetched.")
    results = os.path.join(REPO_ROOT, "results")
    os.makedirs(results, exist_ok=True)
    local_archive = os.path.join(tempfile.gettempdir(), "railway_stations_results.tar.gz")
    if not args.skip_pack:
        print(f"Packing {len(stations)} stations on Roihu: {' '.join(stations)}", flush=True)
        ssh = subprocess.run([SSH, HOST, fetch_remote_command(stations)])  # console, no capture
        if ssh.returncode:
            print(f"ERROR: packing on Roihu failed (ssh exit {ssh.returncode}). If the connection "
                  "closed after the passphrase, run the printed command by hand and then this "
                  "script with --skip-pack:")
            print(f"  ssh {HOST} '{fetch_remote_command(stations)}'")
            return 1
    print("Downloading...", flush=True)
    scp = subprocess.run([SCP, f"{HOST}:{DOWNLOAD}", local_archive])
    if scp.returncode:
        print(f"ERROR: scp exited {scp.returncode}; nothing was unpacked.")
        return 1
    kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(local_archive, mode="r:gz") as archive:
        archive.extractall(results, **kwargs)
    os.remove(local_archive)

    scenarios_file = os.environ.get("COLUMNS_FILE", os.path.join(REPO_ROOT, "config", "scenarios.txt"))
    report = []
    for st in stations:
        folder = os.path.join(results, st)
        if not os.path.isdir(folder):
            report.append(f"{st}: nothing fetched")
            continue
        left = _sort_with_retries(folder, scenarios_file)
        print(f"\n==================== {st} ====================")
        grid_summary.main([folder])
        n_scen, n_json = _count(folder)
        where = "not archived"
        if not args.no_archive and left == 0 and n_json:
            target = os.path.join(args.archive, archive_folder_name(st, all_stations), "results")
            shutil.copytree(folder, target, dirs_exist_ok=True)
            where = target
        elif not args.no_archive:
            where = "NOT archived (unsorted folders or no results)"
        report.append(f"{st}: {n_scen} scenario folders, {n_json} result JSON, {left} unsorted -> {where}")
    print("\n==================== SUMMARY ====================")
    print("\n".join(report))
    print("A full grid of 13 scenarios x 4 models is 13 scenario folders and 52 result JSON.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_stage = sub.add_parser("stage")
    p_stage.add_argument("--stations", default="all")
    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--stations", default="all")
    p_fetch.add_argument("--archive", default=DEFAULT_ARCHIVE)
    p_fetch.add_argument("--no-archive", action="store_true")
    p_fetch.add_argument("--skip-pack", action="store_true",
                         help="The results tar was already made on Roihu; only download it.")
    args = parser.parse_args(argv)
    return stage(args) if args.command == "stage" else fetch(args)


if __name__ == "__main__":
    sys.exit(main())
