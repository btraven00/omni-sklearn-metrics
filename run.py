#!/usr/bin/env python3
"""Run catalog metrics by id: catalog.yaml is the implementation, this file only resolves it.

  run.py --output metrics_jsonl=/abs/out.jsonl --metric ari,nmi \
         --bind labels_true=ref_tsv,labels_pred=clusters_tsv --ref_tsv a.tsv --clusters_tsv b.tsv [c.tsv ...]

Contract (omnibenchmark api >= 0.8, named outputs, design 009 §3.2):
  --output <id>=<absolute path>   the one declared output; written there
  --<input id> <path> [...]       inputs, by the plan's input ids; a gather passes one per member
  --metric ids, --bind arg=input_id,...  which catalog entries, and which plan input feeds each
                                  function argument (unbound: the input named like the argument)
  --output_dir, --name            still passed by the runtime until api 0.10; accepted, ignored

Checks, in the order they run (each error names the metric, argument and file):
  1. catalog    every entry's call, args, kwargs, data types and rules exist   (load_catalog)
  2. pre-flight every requested id and binding; every bound file's header        (plan)
  3. inputs     the data type's rules on each file, once, when it is loaded       (Inputs.get)
  4. metric     the entry's `requires` rules; scores classes cover the labels     (evaluate)
  5. result     a value outside the entry's range fails; NaN is written as null   (evaluate)

Fan-in: an argument bound to several paths runs the metric once per path (once per combination, if
several arguments fan out), in parallel over --workers processes. Writes one jsonl record per
metric and combination (sc-brrr specs/README.md); subject is the directory of each fanned-out
input, or of every input when nothing fans out.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CATALOG = Path(__file__).resolve().parent / "catalog.yaml"
ALLOWED = ("sklearn.metrics.",)  # a catalog entry is code: only these modules may be called
RANGE_TOLERANCE = 1e-9  # float slack on a declared range bound (r2 can come out as 1 + 1e-16)


# --- Rules: name -> check(values) returning a problem description, or None if fine ------------
# Named in catalog.yaml (`types.<t>.rules` and an entry's `requires`); same vocabulary as
# sc-brrr specs/types.yaml where the two overlap.

def _count(mask, what):
    n = int(np.count_nonzero(mask))
    return f"{n} value(s) {what}" if n else None


def _probabilities(v):
    if v.ndim != 2:
        return "not a matrix of class probabilities"
    sums = v.sum(axis=1)
    out_of_unit = _count((v < 0) | (v > 1), "outside [0, 1]")
    unnormalized = _count(~np.isclose(sums, 1), "rows not summing to 1")  # sklearn's own tolerance
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
# Every data type is a TSV whose first column is cell_id. A reader returns
# (cell ids, values as a 1-D or 2-D array, kwargs the data implies for the call).

def read_header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f, delimiter="\t"), None)


def read_table(path):
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        columns = next(r)[1:]
        rows = list(r)
    if not rows:
        raise ValueError(f"{path}: no cells")
    return columns, [x[0] for x in rows], [x[1:] for x in rows]


def _numeric(rows, path):
    try:
        return np.array(rows, dtype=float)
    except ValueError as e:  # a non-number, or rows of different lengths
        raise ValueError(f"{path}: not a numeric table ({e})") from None


def read_labels(path):
    _, ids, rows = read_table(path)
    return ids, np.array([x[0] for x in rows]), {}


def read_values(path):
    _, ids, rows = read_table(path)
    return ids, _numeric(rows, path)[:, 0], {}


def read_embedding(path):
    # ponytail: csv + float() is ~20 s at 1M x 50; a binary embedding format if that starts to matter
    _, ids, rows = read_table(path)
    return ids, _numeric(rows, path), {}


def read_scores(path):
    """Class columns sorted by name (sklearn's class order), passed on as labels=."""
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
    types: dict  # data type -> {columns, rules}
    sha256: str  # of catalog.yaml; on every record


def load_catalog(path=CATALOG):
    """The catalog, after checking it is internally consistent (no data involved)."""
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    types, entries = doc["types"], doc["metrics"]

    ids = [e["id"] for e in entries]
    problems = [f"duplicate metric id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for t, spec in types.items():
        if t not in READERS:
            problems.append(f"type {t!r} has no reader")
        problems += [f"type {t!r}: unknown rule {r!r}" for r in spec["rules"] if r not in RULES]
        if not isinstance(spec["columns"], int) and not str(spec["columns"]).startswith(">="):
            problems.append(f"type {t!r}: columns must be a count or '>=k', got {spec['columns']!r}")
    for e in entries:
        problems += [f"{e['id']}: unknown data type {t!r}" for t in e["args"].values() if t not in types]
        for arg, rules in e.get("requires", {}).items():
            if arg not in e["args"]:
                problems.append(f"{e['id']}: requires names {arg!r}, not one of its args")
            problems += [f"{e['id']}: unknown rule {r!r}" for r in rules if r not in RULES]
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return Catalog({e["id"]: e for e in entries}, types, hashlib.sha256(raw).hexdigest())


def resolve(entry):
    """The callable for a catalog entry, after checking it is allowed and its args fit its signature."""
    call = entry["call"]
    if not call.startswith(ALLOWED):
        raise ValueError(f"{entry['id']}: call {call!r} is outside {ALLOWED}")
    module, _, name = call.rpartition(".")
    f = getattr(importlib.import_module(module), name)
    params = inspect.signature(f).parameters
    for arg in [*entry["args"], *entry.get("kwargs", {})]:
        if arg not in params:
            raise ValueError(f"{entry['id']}: {call} has no argument {arg!r}")
    return f


# --- 2. Pre-flight: requested ids, bindings and every file's header, before any data is read ----

def check_header(path, dtype, types):
    if not os.path.isfile(path):
        raise ValueError(f"{path} ({dtype}): no such file")
    head = read_header(path)
    if not head or head[0] != "cell_id":
        raise ValueError(f"{path} ({dtype}): first column must be cell_id, got {head}")
    spec = types[dtype]["columns"]
    if not columns_ok(spec, len(head) - 1):
        raise ValueError(f"{path} ({dtype}): needs {spec} column(s) after cell_id, got {head[1:]}")


def plan(metric_ids, bind, inputs, catalog):
    """[(entry, fn, {arg: [paths]})] for the requested ids, after the pre-flight."""
    unknown = [m for m in metric_ids if m not in catalog.metrics]
    if unknown:
        raise ValueError(f"unknown metric ids {unknown}; catalog has {sorted(catalog.metrics)}")
    jobs, files = [], set()
    for m in metric_ids:
        e = catalog.metrics[m]
        paths = {}
        for arg, dtype in e["args"].items():
            src = bind.get(arg, arg)
            if src not in inputs:
                raise ValueError(f"{m}: argument {arg} is fed by input {src!r}, which was not given (have {sorted(inputs)}; see --bind)")
            paths[arg] = inputs[src]
            files |= {(path, dtype) for path in inputs[src]}
        jobs.append((e, resolve(e), paths))
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
        """(aligned values, implied kwargs) of a file read as data type dtype."""
        key = (dtype, path)
        if key not in self.cache:
            read, encode = READERS[dtype]
            ids, values, implied = read(path)
            aligned = self._align(ids, values, path)
            apply_rules(aligned, self.types[dtype]["rules"], f"{path} ({dtype})")
            self.cache[key] = (encode(aligned) if encode else aligned, implied)
        return self.cache[key]

    def _align(self, ids, values, path):
        if self.order is None:
            self.order, self.first = {c: i for i, c in enumerate(sorted(ids))}, path
        if len(ids) != len(self.order) or len(set(ids)) != len(ids) or any(c not in self.order for c in ids):
            raise ValueError(f"{path}: cell ids differ from {self.first}'s ({len(ids)} vs {len(self.order)} cells)")
        aligned = np.empty_like(values)
        aligned[[self.order[c] for c in ids]] = values  # joined on cell_id, never on row order
        return aligned


# --- 4-5. Evaluate one metric on one input combination ----------------------------------------

def check_classes(entry, args, implied):
    """The classes of a scores argument must include every label of the entry's labels arguments."""
    if "labels" not in implied:
        return
    classes = set(implied["labels"])
    for arg, dtype in entry["args"].items():
        if dtype == "labels" and (missing := {str(c) for c in np.unique(args[arg])} - classes):
            raise ValueError(f"{entry['id']}: argument {arg} has labels {sorted(missing)} with no column in the scores file")


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
    args, implied = {}, {}
    for arg, path in chosen.items():
        args[arg], extra = data.get(entry["args"][arg], path)
        implied.update({k: v for k, v in extra.items() if k in params})  # a scores file's labels=
    for arg, rules in entry.get("requires", {}).items():
        apply_rules(args[arg], rules, f"{entry['id']}: argument {arg} ({chosen[arg]})")
    check_classes(entry, args, implied)
    return check_result(entry, float(f(**args, **{**implied, **entry.get("kwargs", {})})))


# --- Running: combinations in parallel ---------------------------------------------------------

_DATA = None  # per process: the reference (and anything shared) is read once per worker


def _work(task):
    """All metrics of one input combination, in one process. Returns their records."""
    global _DATA
    import sklearn

    sha, types, group = task
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
    return records


def run(jobs, catalog, workers=1):
    """Records of every metric x input combination; combinations run in parallel, one per task.

    A task aligns its own files, so each worker holds its own copy of shared inputs.
    ponytail: per-member parallelism, each worker holds the reference (~150 MB at 1M cells);
    share it via shared memory if workers x RAM becomes the limit.
    """
    global _DATA
    groups = {}
    for e, f, paths in jobs:
        args = list(paths)
        fanned = [p for p in args if len(paths[p]) > 1]
        for combo in itertools.product(*(paths[p] for p in args)):
            chosen = dict(zip(args, combo))
            subject = " | ".join(os.path.dirname(chosen[p]) for p in (fanned or args))
            groups.setdefault(tuple(sorted(chosen.items())), []).append((e, f, chosen, subject))
    tasks = [(catalog.sha256, catalog.types, g) for g in groups.values()]
    _DATA = None  # a fresh cache per run in this process (pool workers start fresh anyway)
    if workers <= 1 or len(tasks) <= 1:
        results = list(map(_work, tasks))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(workers, len(tasks))) as pool:
            results = list(pool.map(_work, tasks))  # in task order: output is deterministic
    return [r for recs in results for r in recs]


# --- CLI ---------------------------------------------------------------------------------------

def parse_pairs(s):
    """'a=x,b=y' -> {'a': 'x', 'b': 'y'}."""
    return dict(kv.split("=", 1) for kv in s.split(",") if kv) if s else {}


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
    empty = [k for k, v in out.items() if not v]
    if empty:
        raise ValueError(f"inputs without a path: {empty}")
    return out


def parse_output(values):
    """The one --output id=PATH (api >= 0.8) -> its absolute Path."""
    if len(values) != 1:
        raise ValueError(f"expected exactly one --output id=PATH (the stage's metrics output), got {values}")
    oid, sep, path = values[0].partition("=")
    if not sep or not oid or not path:
        raise ValueError(f"--output must be id=PATH, got {values[0]!r}")
    if not os.path.isabs(path):
        raise ValueError(f"--output {oid}: path must be absolute (api >= 0.8), got {path!r}")
    return Path(path)


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)  # else an input id like --m is taken as --metric
    p.add_argument("--output", action="append", default=[], help="id=PATH, absolute (api >= 0.8)")
    p.add_argument("--metric", required=True, help="comma-separated catalog ids")
    p.add_argument("--bind", default="", help="arg=input_id,... (function argument = plan input id)")
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)),
                   help="processes over input combinations (default: the CPUs this process may use)")
    for legacy in ("--output_dir", "--name"):  # passed by the runtime until api 0.10; unused
        p.add_argument(legacy, help=argparse.SUPPRESS)
    args, rest = p.parse_known_args(argv)

    out = parse_output(args.output)
    catalog = load_catalog()
    jobs = plan(args.metric.split(","), parse_pairs(args.bind), parse_inputs(rest), catalog)
    records = run(jobs, catalog, args.workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)


if __name__ == "__main__":
    main()
