#!/usr/bin/env python3
"""Run catalog metrics by id: catalog.yaml is the implementation, this file only resolves it.

  run.py --output metrics_jsonl=/abs/out.jsonl --metric ari,nmi \
         --bind labels_true=ref_tsv,labels_pred=clusters_tsv --ref_tsv a.tsv --clusters_tsv b.tsv [c.tsv ...]

Contract (omnibenchmark api >= 0.8, named outputs, design 009 §3.2):
  --output <id>=<absolute path>   the one declared output; written there
  --<input id> <path> [...]       inputs, by the plan's input ids; a gather passes one per member
  --metric ids                    catalog entries to run (each at most once)
  --bind arg=input_id,...         which plan input feeds each function argument
                                  (unbound: the input named like the argument)
  --pairing zip|product           when several arguments get several files: zip pairs the n-th
                                  files (members of one gather group; default), product pairs
                                  every file with every other (e.g. replicate-to-replicate)
  --workers N                     processes (default: the CPU budget, at most DEFAULT_MAX_WORKERS)
  --output_dir, --name            still passed by the runtime until api 0.10; accepted, ignored

Checks, in the order they run (each error names the metric, argument and file):
  1. catalog    every entry's function, args, kwargs, data types, rules, range   (load_catalog)
  2. pre-flight requested ids, bindings, pairing; every bound file's header      (plan)
  3. inputs     row widths, cell ids, the data type's rules; once per file        (Inputs.get)
  4. metric     the entry's `requires` rules; scores columns match the labels     (evaluate)
  5. result     a value outside the entry's range fails; NaN is written as null   (evaluate)

Writes one jsonl record per metric and input combination (sc-brrr specs/README.md). Its subject
is the directory of the file fed to the entry's last argument (the prediction: labels_pred,
y_pred, the clusters of a silhouette), relative to the omnibenchmark output root (the nearest
ancestor of --output holding .metadata/), or absolute outside one.
"""

import argparse
import csv
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CATALOG = Path(__file__).resolve().parent / "catalog.yaml"
ALLOWED = ("sklearn.metrics.",)  # a catalog entry is code: only these modules may be called
RANGE_TOLERANCE = 1e-9  # float slack on a declared range bound (r2 can come out as 1 + 1e-16)
PROBABILITY_RTOL = float(np.sqrt(np.finfo(float).eps))  # sklearn 1.7's rows-sum-to-1 tolerance
DEFAULT_MAX_WORKERS = 8  # each worker holds its own copy of shared inputs (the reference)
FLAGS = {"output", "metric", "bind", "pairing", "workers", "output_dir", "name"}  # never input ids


# --- Rules: name -> check(values) returning a problem description, or None if fine ------------
# Named in catalog.yaml (`types.<t>.rules` and an entry's `requires`); same vocabulary as
# sc-brrr specs/types.yaml where the two overlap.

def _count(mask, what):
    n = int(np.count_nonzero(mask))
    return f"{n} value(s) {what}" if n else None


def _probabilities(v):
    if v.ndim != 2:
        return "not a matrix of class probabilities"
    out_of_unit = _count((v < 0) | (v > 1), "outside [0, 1]")
    unnormalized = _count(~np.isclose(v.sum(axis=1), 1, rtol=PROBABILITY_RTOL),
                          "rows not summing to 1 (sklearn's tolerance, ~2.5e-8; renormalize the rows)")
    return out_of_unit or unnormalized


def _several_groups(v):
    k, n = len(np.unique(v)), len(v)
    return None if 2 <= k <= n - 1 else f"{k} distinct labels for {n} cells; needs 2 to n-1"


RULES = {
    "finite": lambda v: _count(~np.isfinite(v), "NaN or inf"),
    "positive": lambda v: _count(v <= 0, "<= 0"),
    "nonnegative": lambda v: _count(v < 0, "< 0"),
    "above_minus_one": lambda v: _count(v <= -1, "<= -1"),
    "probabilities": _probabilities,
    "several_groups": _several_groups,
}


def apply_rules(values, rules, where):
    """Raise naming `where` and every failed rule."""
    problems = [f"{r}: {msg}" for r in rules if (msg := RULES[r](values))]
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems))


# --- Data types: reading -----------------------------------------------------------------------
# Every data type is a TSV whose first column is cell_id (a UTF-8 BOM is tolerated). A reader
# returns (cell ids, values as a 1-D or 2-D array, what the file implies for the call).

def read_header(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f, delimiter="\t"), None)


def read_table(path):
    """(column names after cell_id, cell ids, rows of the other columns); every row as wide as the header."""
    # ponytail: Python lists of strings, ~11x the array in memory (~4 GB at 1M x 50); a binary
    # embedding format or a C parser when 1M-cell embeddings are routine
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        rows = list(r)
    for line, row in enumerate(rows, start=2):
        if len(row) != len(header):
            what = "a blank line" if not row else f"{len(row)} field(s), the header has {len(header)}"
            raise ValueError(f"{path}, line {line}: {what}")
    if not rows:
        raise ValueError(f"{path}: no cells")
    return header[1:], [x[0] for x in rows], [x[1:] for x in rows]


def _numeric(rows, path):
    try:
        return np.array(rows, dtype=float)
    except ValueError as e:
        raise ValueError(f"{path}: not a numeric table ({e})") from None


def read_labels(path):
    _, ids, rows = read_table(path)
    return ids, np.array([x[0] for x in rows]), {}


def read_values(path):
    _, ids, rows = read_table(path)
    return ids, _numeric(rows, path)[:, 0], {}


def read_embedding(path):
    _, ids, rows = read_table(path)
    return ids, _numeric(rows, path), {}


def read_scores(path):
    """Class columns sorted by name (sklearn's class order); the names are passed as labels=."""
    classes, ids, rows = read_table(path)
    order = np.argsort(np.array(classes))
    return ids, _numeric(rows, path)[:, order], {"labels": [classes[i] for i in order]}


def code_partition(labels):
    """Labels -> int codes: partition metrics only see which cells share a label."""
    return np.unique(labels, return_inverse=True)[1]


READERS = {  # data type -> (reader, encoder applied once after alignment, or None)
    "partition": (read_labels, code_partition),
    "labels": (read_labels, None),
    "values": (read_values, None),
    "embedding": (read_embedding, None),
    "scores": (read_scores, None),
}


def columns_ok(spec, n):
    """`columns` of a data type: an exact count, or ">=k"."""
    if isinstance(spec, int):
        return n == spec
    return n >= int(spec.removeprefix(">="))


# --- 1. Catalog ---------------------------------------------------------------------------------

@dataclass
class Catalog:
    metrics: dict  # id -> entry
    functions: dict  # id -> the resolved sklearn function
    types: dict  # data type -> {columns, rules}
    sha256: str  # of catalog.yaml; on every record


def resolve(entry):
    """The callable for a catalog entry, after checking it is allowed and its args and kwargs fit its signature."""
    call = entry["call"]
    if not call.startswith(ALLOWED):
        raise ValueError(f"{entry['id']}: call {call!r} is outside {ALLOWED}")
    module, _, name = call.rpartition(".")
    f = getattr(importlib.import_module(module), name, None)
    if f is None:
        raise ValueError(f"{entry['id']}: {call} does not exist")
    params = inspect.signature(f).parameters
    for arg in [*entry["args"], *entry.get("kwargs", {})]:
        if arg not in params:
            raise ValueError(f"{entry['id']}: {call} has no argument {arg!r}")
    return f


def _entry_problems(e, types):
    missing = [k for k in ("id", "call", "args", "higher_is_better", "range") if k not in e]
    if missing:
        return [f"{e.get('id', e)}: missing {missing}"]
    problems = [f"{e['id']}: unknown data type {t!r}" for t in e["args"].values() if t not in types]
    for arg, rules in e.get("requires", {}).items():
        if arg not in e["args"]:
            problems.append(f"{e['id']}: requires names {arg!r}, not one of its args")
        problems += [f"{e['id']}: unknown rule {r!r}" for r in rules if r not in RULES]
    if not isinstance(e["higher_is_better"], bool):
        problems.append(f"{e['id']}: higher_is_better must be true or false")
    r = e["range"]
    bounds_ok = isinstance(r, list) and len(r) == 2 and all(b is None or isinstance(b, (int, float)) for b in r)
    if not bounds_ok or (None not in r and r[0] > r[1]):
        problems.append(f"{e['id']}: range must be [min, max] (null for unbounded), got {r!r}")
    try:
        resolve(e)
    except ValueError as err:
        problems.append(str(err))
    return problems


def load_catalog(path=CATALOG):
    """The catalog, after checking it is internally consistent (no data involved)."""
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    types, entries = doc["types"], doc["metrics"]

    ids = [e.get("id") for e in entries]
    problems = [f"duplicate metric id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for t, spec in types.items():
        if t not in READERS:
            problems.append(f"type {t!r} has no reader")
        problems += [f"type {t!r}: unknown rule {r!r}" for r in spec["rules"] if r not in RULES]
        if not isinstance(spec["columns"], int) and not str(spec["columns"]).startswith(">="):
            problems.append(f"type {t!r}: columns must be a count or '>=k', got {spec['columns']!r}")
    for e in entries:
        problems += _entry_problems(e, types)
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    metrics = {e["id"]: e for e in entries}
    return Catalog(metrics, {i: resolve(e) for i, e in metrics.items()}, types, hashlib.sha256(raw).hexdigest())


# --- 2. Pre-flight: requests, bindings, pairing and every file's header, before any data is read

def check_header(path, dtype, types):
    if not os.path.isfile(path):
        raise ValueError(f"{path} ({dtype}): no such file")
    head = read_header(path)
    if not head or head[0] != "cell_id":
        raise ValueError(f"{path} ({dtype}): first column must be cell_id, got {head}")
    spec = types[dtype]["columns"]
    if not columns_ok(spec, len(head) - 1):
        raise ValueError(f"{path} ({dtype}): needs {spec} column(s) after cell_id, got {head[1:]}")
    if dup := sorted({c for c in head if head.count(c) > 1}):
        raise ValueError(f"{path} ({dtype}): duplicate column(s) {dup}")


def combinations(entry, paths, pairing):
    """[{arg: path}]: the input combinations one entry runs on.

    One file per argument: one combination. Several arguments with several files: zip pairs the
    n-th files (they must be equally many), product every file with every other, except a file
    with itself.
    """
    fanned = {arg: len(ps) for arg, ps in paths.items() if len(ps) > 1}
    if pairing == "zip":
        if len(set(fanned.values())) > 1:
            raise ValueError(f"{entry['id']}: --pairing zip needs equally many files per argument, got {fanned}")
        lists = [tuple(paths[arg]) for arg in fanned]
        if len(set(lists)) < len(lists):
            raise ValueError(f"{entry['id']}: arguments {sorted(fanned)} get the same files, so zip would pair "
                             f"each file with itself; use --pairing product to compare them with each other")
        n = max(fanned.values(), default=1)
        return [{arg: ps[i] if len(ps) > 1 else ps[0] for arg, ps in paths.items()} for i in range(n)]
    combos = [dict(zip(paths, c)) for c in itertools.product(*paths.values())]
    return [c for c in combos if len({c[arg] for arg in fanned}) == len(fanned)]


def plan(metric_ids, bind, inputs, catalog, pairing="zip"):
    """[(entry, fn, [{arg: path}])] for the requested ids, after the pre-flight."""
    if dup := sorted({m for m in metric_ids if metric_ids.count(m) > 1}):
        raise ValueError(f"--metric lists {dup} more than once")
    if unknown := [m for m in metric_ids if m not in catalog.metrics]:
        raise ValueError(f"unknown metric ids {unknown}; catalog has {sorted(catalog.metrics)}")
    entries = [catalog.metrics[m] for m in metric_ids]
    known_args = {arg for e in entries for arg in e["args"]}
    if stray := sorted(set(bind) - known_args):
        raise ValueError(f"--bind names {stray}, which no requested metric takes (they take {sorted(known_args)})")

    jobs, files = [], set()
    for e in entries:
        paths = {}
        for arg, dtype in e["args"].items():
            src = bind.get(arg, arg)
            if src in FLAGS:
                raise ValueError(f"{e['id']}: argument {arg} is fed by input {src!r}, which is also this "
                                 f"module's --{src} flag; rename that input in the plan")
            if src not in inputs:
                raise ValueError(f"{e['id']}: argument {arg} is fed by input {src!r}, which was not given "
                                 f"(have {sorted(inputs)}; see --bind)")
            paths[arg] = inputs[src]
            files |= {(path, dtype) for path in inputs[src]}
        jobs.append((e, catalog.functions[e["id"]], combinations(e, paths, pairing)))
    for path, dtype in sorted(files):
        check_header(path, dtype, catalog.types)
    return jobs


# --- 3. Inputs: each file loaded and checked once ----------------------------------------------

class Inputs:
    """Each file loaded once, its rows in sorted cell_id order: the same order whatever order
    the files list their cells in, so order-sensitive metrics (a sampled silhouette) are too."""

    def __init__(self, types):
        self.types, self.order, self.first, self.cache = types, None, None, {}

    def get(self, dtype, path):
        """(aligned values, what the file implies for the call) of a file read as data type dtype."""
        key = (dtype, path)
        if key not in self.cache:
            read, encode = READERS[dtype]
            ids, values, implied = read(path)
            aligned = self._align(ids, values, path)
            apply_rules(aligned, self.types[dtype]["rules"], f"{path} ({dtype})")
            self.cache[key] = (encode(aligned) if encode else aligned, implied)
        return self.cache[key]

    def keep_only(self, paths):
        """Drop every cached file not in paths (a finished gather member's files)."""
        self.cache = {k: v for k, v in self.cache.items() if k[1] in paths}

    def _align(self, ids, values, path):
        if dup := len(ids) - len(set(ids)):
            raise ValueError(f"{path}: {dup} duplicate cell id(s)")
        if self.order is None:
            self.order, self.first = {c: i for i, c in enumerate(sorted(ids))}, path
        if len(ids) != len(self.order) or any(c not in self.order for c in ids):
            raise ValueError(f"{path}: cell ids differ from {self.first}'s ({len(ids)} vs {len(self.order)} cells)")
        aligned = np.empty_like(values)
        aligned[[self.order[c] for c in ids]] = values  # joined on cell_id, never on row order
        return aligned


# --- 4-5. Evaluate one metric on one input combination ----------------------------------------

def check_classes(entry, args, classes, passed_as_labels):
    """A scores file's columns against the labels arguments.

    Every label needs a column. When the function takes no labels= (average_precision), it
    infers the classes from the labels themselves, so every column also needs a label.
    """
    for arg, dtype in entry["args"].items():
        if dtype != "labels":
            continue
        present = {str(c) for c in np.unique(args[arg])}
        if missing := present - classes:
            raise ValueError(f"{entry['id']}: argument {arg} has labels {sorted(missing)} with no column in the scores file")
        if not passed_as_labels and (unused := classes - present):
            raise ValueError(f"{entry['id']}: the scores file has columns {sorted(unused)} with no cell in argument "
                             f"{arg}; {entry['call']} takes its classes from {arg}, so every column needs one")


def check_result(entry, value):
    """(value, note): NaN and inf are kept as null with a note; out of range is an error."""
    if not math.isfinite(value):
        return None, f"sklearn returned {value}"
    lo, hi = entry["range"]
    if (lo is not None and value < lo - RANGE_TOLERANCE) or (hi is not None and value > hi + RANGE_TOLERANCE):
        raise ValueError(f"{entry['id']}: value {value} outside its declared range {entry['range']}")
    return value, None


def evaluate(entry, f, chosen, data):
    """(value, note) of one entry on one input combination ({arg: path}), with checks 4 and 5."""
    params = inspect.signature(f).parameters
    args, extra = {}, {}
    for arg, path in chosen.items():
        args[arg], implied = data.get(entry["args"][arg], path)
        extra.update(implied)
    for arg, rules in entry.get("requires", {}).items():
        apply_rules(args[arg], rules, f"{entry['id']}: argument {arg} ({chosen[arg]})")
    if "labels" in extra:  # a scores file
        check_classes(entry, args, set(extra["labels"]), passed_as_labels="labels" in params)
    passed = {k: v for k, v in extra.items() if k in params}
    return check_result(entry, float(f(**args, **{**passed, **entry.get("kwargs", {})})))


# --- Running: combinations in parallel ---------------------------------------------------------

_DATA = None  # per process: files shared by several combinations (the reference) are read once


def _limit_threads():
    """In a pool worker: one BLAS/OpenMP thread, so workers x threads stays within the CPU budget."""
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)


def _work(task):
    """All metrics of one input combination, in one process. Returns their records."""
    global _DATA
    import sklearn

    sha, types, shared, group = task
    _DATA = _DATA or Inputs(types)
    records = []
    for e, f, chosen, subject in group:
        value, note = evaluate(e, f, chosen, _DATA)
        attrs = {
            "call": e["call"],
            "kwargs": e.get("kwargs", {}),
            "inputs": chosen,
            "higher_is_better": e["higher_is_better"],
            "range": e["range"],
            "catalog_sha256": sha,
            "sklearn_version": sklearn.__version__,
        }
        if note:
            attrs["note"] = note
        records.append({"metric": e["id"], "value": value, "subject": subject, "rep": None, "attrs": attrs})
    _DATA.keep_only(shared)
    return records


def output_root(path):
    """The omnibenchmark output root above path (the directory holding .metadata/), or None."""
    return next((p for p in Path(path).parents if (p / ".metadata").is_dir()), None)


def subject_of(entry, chosen, root):
    """The directory of the file fed to the entry's last argument, relative to root if given."""
    d = Path(chosen[list(entry["args"])[-1]]).parent
    return str(d.relative_to(root)) if root and d.is_relative_to(root) else str(d)


def run(jobs, catalog, workers=1, root=None):
    """Records of every metric x input combination; combinations run in parallel, one per task."""
    global _DATA
    groups = {}
    for e, f, combos in jobs:
        for chosen in combos:
            groups.setdefault(tuple(sorted(chosen.items())), []).append((e, f, chosen, subject_of(e, chosen, root)))
    uses = Counter(path for key in groups for path in {p for _, p in key})
    shared = {path for path, n in uses.items() if n > 1}  # kept in a worker; the rest is dropped after use
    tasks = [(catalog.sha256, catalog.types, shared, g) for g in groups.values()]
    _DATA = None  # a fresh cache per run in this process (pool workers start fresh anyway)
    if workers <= 1 or len(tasks) <= 1:
        results = list(map(_work, tasks))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(workers, len(tasks)), initializer=_limit_threads) as pool:
            results = list(pool.map(_work, tasks))  # in task order: output is deterministic
    return [r for recs in results for r in recs]


def cpu_budget():
    """CPUs this process may use: its affinity, capped by a cgroup v2 CPU quota (podman --cpus)."""
    n = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


# --- CLI ---------------------------------------------------------------------------------------

def parse_bind(s):
    """'a=x,b=y' -> {'a': 'x', 'b': 'y'}."""
    pairs = {}
    for item in filter(None, s.split(",")):
        arg, sep, src = item.partition("=")
        if not (sep and arg and src):
            raise ValueError(f"--bind: {item!r} is not arg=input_id")
        pairs[arg] = src
    return pairs


def parse_inputs(rest):
    """['--a', 'x', '--b', 'y', 'z'] -> {'a': ['x'], 'b': ['y', 'z']}."""
    out, key = {}, None
    for tok in rest:
        if tok.startswith("--"):
            key = tok[2:]
            out.setdefault(key, [])
        elif key is None:
            raise ValueError(f"path {tok!r} before any --<input id>")
        else:
            out[key].append(tok)
    if empty := [k for k, v in out.items() if not v]:
        raise ValueError(f"inputs without a path: {empty}")
    return out


def parse_output(values):
    """The one --output id=PATH (api >= 0.8) -> its absolute Path."""
    if len(values) != 1:
        raise ValueError(f"expected exactly one --output id=PATH (the stage's metrics output), got {values}")
    oid, sep, path = values[0].partition("=")
    if not (sep and oid and path):
        raise ValueError(f"--output must be id=PATH, got {values[0]!r}")
    if not os.path.isabs(path):
        raise ValueError(f"--output {oid}: path must be absolute (api >= 0.8), got {path!r}")
    return Path(path)


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)  # else an input id like --m is taken as --metric
    p.add_argument("--output", action="append", default=[], help="id=PATH, absolute (api >= 0.8)")
    p.add_argument("--metric", required=True, help="comma-separated catalog ids")
    p.add_argument("--bind", default="", help="arg=input_id,... (function argument = plan input id)")
    p.add_argument("--pairing", choices=["zip", "product"], default="zip",
                   help="several files on several arguments: pair the n-th (zip) or all with all (product)")
    p.add_argument("--workers", type=int, default=min(cpu_budget(), DEFAULT_MAX_WORKERS),
                   help=f"processes over input combinations (default: the CPU budget, at most {DEFAULT_MAX_WORKERS})")
    for legacy in ("--output_dir", "--name"):  # passed by the runtime until api 0.10; unused
        p.add_argument(legacy, help=argparse.SUPPRESS)
    args, rest = p.parse_known_args(argv)

    out = parse_output(args.output)
    catalog = load_catalog()
    jobs = plan(args.metric.split(","), parse_bind(args.bind), parse_inputs(rest), catalog, args.pairing)
    records = run(jobs, catalog, args.workers, output_root(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)


if __name__ == "__main__":
    main()
