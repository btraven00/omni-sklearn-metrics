#!/usr/bin/env python3
"""Run catalog metrics by id: catalog.yaml is the implementation, this file only resolves it.

  run.py --output_dir D --name N --metric ari,nmi --bind labels_true=ref_tsv,labels_pred=clusters_tsv \
         --ref_tsv a.tsv --clusters_tsv b.tsv
  run.py --selftest

Inputs arrive as --<input id> <path> (omnibenchmark passes them by the plan's input ids).
--bind maps each catalog port to one of them; a port with no binding uses the input of the
same name. Writes {name}_metrics.jsonl, one record per metric (sc-brrr specs/README.md).
Every id, call, port and binding is checked before any data is read.
"""

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import yaml

CATALOG = Path(__file__).resolve().parent / "catalog.yaml"
ALLOWED = ("sklearn.metrics.",)  # a catalog entry is code: only these modules may be called


def read_partition(path):
    """A TSV of cell_id plus one label column -> {cell_id: label}."""
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        head = next(r)
        if len(head) != 2 or head[0] != "cell_id":
            raise ValueError(f"{path}: partition needs columns cell_id + one label column, got {head}")
        out = dict(r)
    return out


# ponytail: one port type; add labels / scores / matrix converters when a catalog entry needs them
CONVERTERS = {"partition": read_partition}


def align(inputs):
    """Port -> {cell_id: value} dicts, as arrays in one shared cell order. Never joins on row order."""
    ids = None
    for port, d in inputs.items():
        if ids is None:
            ids, first = sorted(d), port
        elif set(d) != set(ids):
            raise ValueError(f"port {port}: cell ids differ from port {first}'s ({len(d)} vs {len(ids)} cells)")
    return {port: np.array([d[i] for i in ids]) for port, d in inputs.items()}


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
    """[(entry, fn, {port: input path})] for the requested ids; raises before any data is read."""
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

    cache = {}
    for e, f, paths in jobs:
        data = {p: cache.setdefault((e["ports"][p], path), CONVERTERS[e["ports"][p]](path)) for p, path in paths.items()}
        value = float(f(**align(data), **e.get("kwargs", {})))
        yield {
            "metric": e["id"],
            "value": value,
            "subject": name,
            "rep": None,
            "attrs": {
                "call": e["call"],
                "kwargs": e.get("kwargs", {}),
                "inputs": paths,
                "higher_is_better": e["higher_is_better"],
                "range": e["range"],
                "catalog_sha256": sha,
                "sklearn_version": sklearn.__version__,
            },
        }


def pairs(s):
    return dict(kv.split("=", 1) for kv in s.split(",") if kv) if s else {}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--metric", required=True, help="comma-separated catalog ids")
    p.add_argument("--bind", default="", help="port=input_id,...")
    args, rest = p.parse_known_args(argv)
    if len(rest) % 2 or not all(k.startswith("--") for k in rest[::2]):
        p.error(f"inputs must be --<input id> <path> pairs, got {rest}")
    inputs = {k[2:]: v for k, v in zip(rest[::2], rest[1::2])}

    catalog, sha = load_catalog()
    jobs = plan(args.metric.split(","), pairs(args.bind), inputs, catalog)
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
    truth, pred = rng.integers(0, 5, n), rng.integers(0, 7, n)
    pred[:300] = truth[:300]  # partly agreeing, so values are not ~0
    d = Path(tempfile.mkdtemp())
    order = rng.permutation(n)  # the two files list cells in different orders
    for fname, lab, idx in [("t.tsv", truth, range(n)), ("p.tsv", pred, order)]:
        with open(d / fname, "w") as f:
            f.write("cell_id\tcluster\n" + "".join(f"{ids[i]}\t{lab[i]}\n" for i in idx))

    catalog, _ = load_catalog()
    main(["--output_dir", str(d), "--name", "x", "--metric", ",".join(catalog),
          "--bind", "labels_true=truth,labels_pred=pred", "--truth", str(d / "t.tsv"), "--pred", str(d / "p.tsv")])
    recs = [json.loads(l) for l in open(d / "x_metrics.jsonl")]
    assert [r["metric"] for r in recs] == list(catalog)
    for r in recs:
        e = catalog[r["metric"]]
        want = getattr(skm, e["call"].rpartition(".")[2])(truth, pred, **e.get("kwargs", {}))
        assert abs(r["value"] - want) < 1e-12, (r["metric"], r["value"], want)
        lo, hi = e["range"]
        assert (lo is None or r["value"] >= lo) and (hi is None or r["value"] <= hi), r

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
    with open(d / "short.tsv", "w") as f:
        f.write("cell_id\tcluster\nc0\t1\n")
    fails(lambda: align({"a": read_partition(d / "t.tsv"), "b": read_partition(d / "short.tsv")}), "cell ids differ")
    print(f"selftest ok: {len(recs)} metrics match sklearn")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--selftest"] else main()
