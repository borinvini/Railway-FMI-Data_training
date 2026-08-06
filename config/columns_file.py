"""Parse the plain-text feature file passed to `main.py --columns-file`.

A module of its own, not a helper inside const_training.py, because two callers
need it and they must not import each other: main.py validates the file early
(before any pipeline work) and const_training.py performs the actual binding.
Standard library only — const_training.py imports this, so any dependency on
src/ or on another config module would create an import cycle.
"""
from pathlib import Path


def load_columns(path):
    """Return the column names listed in `path`, one per line, in file order.

    Blank lines and lines whose first non-space character is `#` are ignored.
    Names are returned verbatim: they may contain spaces and parentheses, and
    are NOT checked against any dataset — src/training_pipeline.py:1350 already
    validates them against the real DataFrame and names the missing ones.

    Raises FileNotFoundError if `path` does not exist, and ValueError if the
    file yields no names or lists the same name twice.
    """
    # utf-8-sig transparently strips the byte-order mark that Windows editors
    # prepend, which would otherwise become part of the first column name.
    text = Path(path).read_text(encoding="utf-8-sig")

    columns = []
    first_seen = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        # .strip() also removes the trailing \r left by copying a CRLF file
        # from Windows to Linux, which would corrupt every name in the file.
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        if name in first_seen:
            raise ValueError(
                f"{path}: duplicate column {name!r} on lines "
                f"{first_seen[name]} and {lineno}"
            )
        first_seen[name] = lineno
        columns.append(name)

    if not columns:
        raise ValueError(
            f"{path}: no column names found. An empty selection would drop "
            f"select_training_cols into its interactive prompt, which hangs "
            f"under Slurm (stdin is /dev/null)."
        )

    return columns
