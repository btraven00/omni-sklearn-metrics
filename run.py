#!/usr/bin/env python3
"""Run catalog metrics by id: catalog.yaml is the implementation, this file only resolves it.

  run.py --output_dir D --name N --metric ari,nmi --bind labels_true=ref_tsv,labels_pred=clusters_tsv \
         --ref_tsv a.tsv --clusters_tsv b.tsv [c.tsv ...]
  run.py --selftest

Inputs arrive as --<input id> <path> [<path> ...] (omnibenchmark passes them by the plan's input
ids; a gather passes one path per member). --bind maps each catalog port to one of them; a port
with no binding uses the input of the same name.

Fan-in: a port bound to several paths runs the metric once per path (once per combination, if
several ports fan out). Every file is read, joined on cell_id and coded once, into one cell
order shared by all files, so a pair costs only the sklearn call.

Writes {name}_metrics.jsonl, one record per metric and combination (sc-brrr specs/README.md);
subject is --name, or with fan-in the directory of each fanned-out file.
Every id, call, port and binding is checked before any data is read.
"""

import argparse
import csv
import hashlib
import importlib
import inspect
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

CATALOG = Path(__file__).resolve().parent / "catalog.yaml"
ALLOWED = ("sklearn.metrics.",)  # a catalog entry is code: only these modules may be called


def read_partition(path):
    """A TSV of cell_id plus one label column -> (cell ids, labels)."""
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        head = next(r)
        if len(head) != 2 or head[0] != "cell_id":
            raise ValueError(f"{path}: partition needs columns cell_id + one label column, got {head}")
        rows = list(r)
    if not rows:
        raise ValueError(f"{path}: no cells")
    return [x[0] for x in rows], np.array([x[1] for x in rows])


def code_partition(labels):
    """Labels -> int codes: partition metrics only see which cells share a label."""
    return np.unique(labels, return_inverse=True)[1]


# port type -> (reader: path -> (ids, values), encoder applied once after alignment)
# ponytail: one port type; add labels / scores / matrix when a catalog entry needs them
CONVERTERS = {"partition": (read_partition, code_partition)}


class Inputs:
    """Each file loaded once, as an array in one cell order shared by every file of the run."""

    def __init__(self):
        self.order, self.first, self.cache = None, None, {}

    def get(self, ptype, path):
        key = (ptype, path)
        if key not in self.cache:
            read, encode = CONVERTERS[ptype]
            ids, values = read(path)
            if self.order is None:
                self.order, self.first = {c: i for i, c in enumerate(ids)}, path
            if len(ids) != len(self.order) or len(set(ids)) != len(ids) or any(c not in self.order for c in ids):
                raise ValueError(f"{path}: cell ids differ from {self.first}'s ({len(ids)} vs {len(self.order)} cells)")
            aligned = np.empty_like(values)
            aligned[[self.order[c] for c in ids]] = values  # joined on cell_id, never on row order
            self.cache[key] = encode(aligned)
        return self.cache[key]


def load_catalog(path=CATALOG):
    raw = path.read_bytes()
    entries = {m["id"]: m for m in yaml.safe_load(raw)["metrics"]}
    return entries, hashlib.sha256(raw).hexdigest()


def resolve(entry):
    """The callable for a catalog entry, after checking it is allowed and its ports fit its signature."""
    call = entry["call"]
    if not call.startswith(ALLOWED):
        raise ValueError(f"{entry['id']}: call {call!r} is outside {ALLOWED}")
    mod, _, fn = call.rpartition(".")
    f = getattr(importlib.import_module(mod), fn)
    params = inspect.signature(f).parameters
    for arg in [*entry["ports"], *entry.get("kwargs", {})]:
        if arg not in params:
            raise ValueError(f"{entry['id']}: {call} has no argument {arg!r}")
    for t in entry["ports"].values():
        if t not in CONVERTERS:
            raise ValueError(f"{entry['id']}: no converter for port type {t!r}")
    return f


def plan(metric_ids, bind, inputs, catalog):
    """[(entry, fn, {port: [paths]})] for the requested ids; raises before any data is read."""
    unknown = [m for m in metric_ids if m not in catalog]
    if unknown:
        raise ValueError(f"unknown metric ids {unknown}; catalog has {sorted(catalog)}")
    jobs = []
    for m in metric_ids:
        e = catalog[m]
        paths = {}
        for port in e["ports"]:
            src = bind.get(port, port)
            if src not in inputs:
                raise ValueError(f"{m}: port {port} bound to input {src!r}, not given (have {sorted(inputs)})")
            paths[port] = inputs[src]
        jobs.append((e, resolve(e), paths))
    return jobs


def run(jobs, name, sha):
    import sklearn

    data = Inputs()
    for e, f, paths in jobs:
        ports = list(paths)
        fanned = [p for p in ports if len(paths[p]) > 1]
        for combo in itertools.product(*(paths[p] for p in ports)):
            chosen = dict(zip(ports, combo))
            args = {p: data.get(e["ports"][p], chosen[p]) for p in ports}
            yield {
                "metric": e["id"],
                "value": float(f(**args, **e.get("kwargs", {}))),
                "subject": " | ".join(os.path.dirname(chosen[p]) for p in fanned) or name,
                "rep": None,
                "attrs": {
                    "call": e["call"],
                    "kwargs": e.get("kwargs", {}),
                    "inputs": chosen,
                    "higher_is_better": e["higher_is_better"],
                    "range": e["range"],
                    "catalog_sha256": sha,
                    "sklearn_version": sklearn.__version__,
                },
            }


def pairs(s):
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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--metric", required=True, help="comma-separated catalog ids")
    p.add_argument("--bind", default="", help="port=input_id,...")
    args, rest = p.parse_known_args(argv)

    catalog, sha = load_catalog()
    jobs = plan(args.metric.split(","), pairs(args.bind), parse_inputs(rest), catalog)
    out = Path(args.output_dir) / f"{args.name}_metrics.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for rec in run(jobs, args.name, sha):
            fh.write(json.dumps(rec) + "\n")


def selftest():
    import sklearn.metrics as skm
    import tempfile

    rng = np.random.default_rng(0)
    n = 500
    ids = [f"c{i}" for i in range(n)]
    truth = rng.integers(0, 5, n)
    d = Path(tempfile.mkdtemp())

    def write(path, lab):
        path.parent.mkdir(parents=True, exist_ok=True)
        idx = rng.permutation(n)  # every file lists cells in its own order
        with open(path, "w") as f:
            f.write("cell_id\tcluster\n" + "".join(f"{ids[i]}\tk{lab[i]}\n" for i in idx))

    write(d / "t.tsv", truth)
    preds = []
    for m in range(3):  # three members of a gather, agreeing with truth to different degrees
        lab = rng.integers(0, 7, n)
        lab[: 100 * (m + 1)] = truth[: 100 * (m + 1)]
        preds.append(lab)
        write(d / f"m{m}" / "p.tsv", lab)

    catalog, _ = load_catalog()
    main(["--output_dir", str(d), "--name", "x", "--metric", ",".join(catalog),
          "--bind", "labels_true=truth,labels_pred=pred", "--truth", str(d / "t.tsv"),
          "--pred", *(str(d / f"m{m}" / "p.tsv") for m in range(3))])
    recs = [json.loads(l) for l in open(d / "x_metrics.jsonl")]
    assert len(recs) == 3 * len(catalog), len(recs)
    for r in recs:
        e = catalog[r["metric"]]
        m = int(Path(r["subject"]).name[1:])
        assert r["subject"] == str(d / f"m{m}"), r["subject"]
        want = getattr(skm, e["call"].rpartition(".")[2])(truth, preds[m], **e.get("kwargs", {}))
        assert abs(r["value"] - want) < 1e-12, (r["metric"], m, r["value"], want)
        lo, hi = e["range"]
        assert (lo is None or r["value"] >= lo) and (hi is None or r["value"] <= hi), r

    main(["--output_dir", str(d), "--name", "one", "--metric", "ari",  # no fan-in: subject is --name
          "--labels_true", str(d / "t.tsv"), "--labels_pred", str(d / "m0" / "p.tsv")])
    assert json.loads(open(d / "one_metrics.jsonl").readline())["subject"] == "one"

    def fails(fn, msg):
        try:
            fn()
        except ValueError as ex:
            assert msg in str(ex), ex
        else:
            raise AssertionError(f"expected ValueError with {msg!r}")

    fails(lambda: plan(["nope"], {}, {}, catalog), "unknown metric ids")
    fails(lambda: plan(["ari"], {}, {}, catalog), "not given")
    fails(lambda: resolve({"id": "evil", "call": "os.system", "ports": {}}), "outside")
    fails(lambda: resolve({"id": "bad", "call": "sklearn.metrics.adjusted_rand_score", "ports": {"x": "partition"}}), "no argument")
    fails(lambda: parse_inputs(["--a"]), "without a path")
    with open(d / "short.tsv", "w") as f:
        f.write("cell_id\tcluster\nc0\t1\n")
    inp = Inputs()
    inp.get("partition", str(d / "t.tsv"))
    fails(lambda: inp.get("partition", str(d / "short.tsv")), "cell ids differ")
    print(f"selftest ok: {len(recs)} records (3 members x {len(catalog)} metrics) match sklearn")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--selftest"] else main()
