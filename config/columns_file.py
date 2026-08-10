"""Parse the plain-text feature file passed to `main.py --columns-file`.

A module of its own, not a helper inside const_training.py, because two callers
need it and they must not import each other: main.py validates the file early
(before any pipeline work) and const_training.py performs the actual binding.
Standard library only — const_training.py imports this, so any dependency on
src/ or on another config module would create an import cycle.

Two file shapes are supported. A plain feature file is one column name per line
(config/features.txt). A scenario file (config/scenarios.txt) holds several such
lists, each opened by a numbered banner comment, so all eight feature sets reach
Roihu in one scp with one sha256.
"""
import re
from pathlib import Path

# A scenario banner: a comment whose first non-space content is a number, a
# dash, and a name — "# 6 - OPERACIONAL + INSTANT WEATHER + ROLLING WINDOWS 72h".
# The `# ====` rule lines and ordinary prose comments do not match.
_BANNER_RE = re.compile(r"^#\s*(\d+)\s*-\s*\S")


def _read_lines(path):
    """Return `path`'s lines with the BOM stripped and line endings normalised.

    utf-8-sig transparently strips the byte-order mark that Windows editors
    prepend, which would otherwise become part of the first column name.
    """
    return Path(path).read_text(encoding="utf-8-sig").splitlines()


def _catalogue(scenarios):
    """Render the available scenarios for an error message.

    Errors name every option because the alternative — "no such scenario" — sends
    the reader back to a gitignored file they may not have open.
    """
    return "\n".join(f"  {i}. {name}" for i, (name, _) in enumerate(scenarios, start=1))


def has_scenarios(path):
    """True when `path` looks like a scenario file rather than a feature file.

    The threshold is two banners, not one: a plain features.txt may legitimately
    carry a single comment such as "# 12h block - see docs", which matches the
    banner shape by accident. Two is enough to distinguish intent, and the real
    scenario file has eight.
    """
    return sum(1 for line in _read_lines(path) if _BANNER_RE.match(line.strip())) >= 2


def load_columns(path):
    """Return the column names listed in `path`, one per line, in file order.

    Blank lines and lines whose first non-space character is `#` are ignored.
    Names are returned verbatim: they may contain spaces and parentheses, and
    are NOT checked against any dataset — src/training_pipeline.py:1350 already
    validates them against the real DataFrame and names the missing ones.

    Raises FileNotFoundError if `path` does not exist, and ValueError if the
    file yields no names, lists the same name twice, or is a scenario file
    (which repeats names across sections by design — use select_scenario).
    """
    lines = _read_lines(path)

    if has_scenarios(path):
        count = sum(1 for line in lines if _BANNER_RE.match(line.strip()))
        raise ValueError(
            f"{path}: contains {count} scenario sections — pass "
            f"--scenario <index|name> to choose one."
        )

    columns = []
    first_seen = {}
    for lineno, raw in enumerate(lines, start=1):
        # _read_lines() already normalises CRLF/CR line endings; .strip()
        # here removes surrounding whitespace and is a backstop for any stray \r.
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


def load_scenarios(path):
    """Return `[(name, [columns]), ...]` for every section in `path`, in file order.

    Duplicate detection is scoped to one section: every scenario legitimately
    repeats trainDelayed, trainStopping and the rest, so the global check in
    load_columns would reject a perfectly valid catalogue.

    Raises ValueError for a duplicate within a section, an empty section, a
    column name appearing before the first banner, or a file with no sections.
    """
    scenarios = []
    name = None
    columns = []
    first_seen = {}

    def close_section():
        if not columns:
            raise ValueError(f"{path}: scenario {name!r} lists no columns.")
        scenarios.append((name, list(columns)))

    for lineno, raw in enumerate(_read_lines(path), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if _BANNER_RE.match(line):
                if name is not None:
                    close_section()
                name = line.lstrip("#").strip()
                columns = []
                first_seen = {}
            continue
        if name is None:
            raise ValueError(
                f"{path}: column {line!r} on line {lineno} appears before any "
                f"scenario banner."
            )
        if line in first_seen:
            raise ValueError(
                f"{path}: duplicate column {line!r} in scenario {name!r} on "
                f"lines {first_seen[line]} and {lineno}"
            )
        first_seen[line] = lineno
        columns.append(line)

    if name is not None:
        close_section()

    if not scenarios:
        raise ValueError(
            f"{path}: no scenario sections found. A section opens with a banner "
            f"comment such as '# 1 - ALL FEATURES'."
        )

    return scenarios


def select_scenario(path, selector):
    """Return `(name, columns)` for one scenario in `path`.

    `selector` is a 1-based index (int or digit string, as argparse delivers it)
    or an exact scenario name. Raises ValueError naming every available scenario
    when it matches nothing.
    """
    scenarios = load_scenarios(path)
    text = str(selector).strip()

    if text.isdigit():
        index = int(text)
        if not 1 <= index <= len(scenarios):
            raise ValueError(
                f"{path}: scenario index {index} is out of range "
                f"1..{len(scenarios)}. Available:\n{_catalogue(scenarios)}"
            )
        return scenarios[index - 1]

    for name, columns in scenarios:
        if name == text:
            return name, columns

    raise ValueError(
        f"{path}: no scenario named {text!r}. Available:\n{_catalogue(scenarios)}"
    )
