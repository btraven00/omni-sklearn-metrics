#!/usr/bin/env python3
"""Run catalog metrics by id: catalog.yaml is the implementation, this file only resolves it.

  run.py --output metrics_jsonl=/abs/out.jsonl --metric ari,nmi \
         --bind labels_true=ref_tsv,labels_pred=clusters_tsv --ref_tsv a.tsv --clusters_tsv b.tsv [c.tsv ...]
  run.py --selftest

Omnibenchmark api >= 0.8 only (named outputs, design 009 §3.2): the one declared output arrives
as --output <id>=<absolute path> and is written there. --output_dir and --name, which the
runtime still passes until api 0.10, are accepted and ignored.

Inputs arrive as --<input id> <path> [<path> ...] (omnibenchmark passes them by the plan's input
ids; a gather passes one path per member). --bind maps each catalog port to one of them; a port
with no binding uses the input of the same name.

Fan-in: a port bound to several paths runs the metric once per path (once per combination, if
several ports fan out), in parallel over --workers processes. Each file is read, joined on
cell_id and coded once per worker, so a pair costs only the sklearn call.

Writes one jsonl record per metric and combination (sc-brrr specs/README.md). subject is the
directory of each fanned-out input, or of every input when nothing fans out.
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


def read_table(path, what, ncols=None):
    """A TSV whose first column is cell_id -> (other column names, cell ids, rows of the other columns)."""
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        head = next(r, None)
        rows = list(r)
    if not head or head[0] != "cell_id" or len(head) < 2 or (ncols and len(head) != ncols + 1):
        raise ValueError(f"{path}: expected columns cell_id + {what}, got {head}")
    if not rows:
        raise ValueError(f"{path}: no cells")
    return head[1:], [x[0] for x in rows], [x[1:] for x in rows]


# Readers: path -> (cell ids, values: 1-D or 2-D array, kwargs the port implies for the call).
def read_labels(path):
    _, ids, rows = read_table(path, "one label column", 1)
    return ids, np.array([x[0] for x in rows]), {}


def read_values(path):
    _, ids, rows = read_table(path, "one numeric column", 1)
    return ids, np.array([x[0] for x in rows], dtype=float), {}


def read_embedding(path):
    # ponytail: csv + float() is ~20 s at 1M x 50; a binary embedding format if that starts to matter
    _, ids, rows = read_table(path, "numeric columns")
    return ids, np.array(rows, dtype=float), {}


def read_scores(path):
    """Class columns sorted by name (sklearn's class order), passed on as labels=."""
    cols, ids, rows = read_table(path, "one probability column per class")
    if len(cols) < 3:
        raise ValueError(f"{path}: scores need >= 3 class columns (binary scores are not supported), got {cols}")
    order = np.argsort(np.array(cols))
    return ids, np.array(rows, dtype=float)[:, order], {"labels": [cols[i] for i in order]}


def code_partition(labels):
    """Labels -> int codes: partition metrics only see which cells share a label."""
    return np.unique(labels, return_inverse=True)[1]


# port type -> (reader, encoder applied once after alignment)
CONVERTERS = {
    "partition": (read_labels, code_partition),
    "labels": (read_labels, None),
    "values": (read_values, None),
    "embedding": (read_embedding, None),
    "scores": (read_scores, None),
}


class Inputs:
    """Each file loaded once, its rows in sorted cell_id order: the same order whatever order
    the files list their cells in, so order-sensitive metrics (a sampled silhouette) are too."""

    def __init__(self):
        self.order, self.first, self.cache = None, None, {}

    def get(self, ptype, path):
        """(aligned values, implied kwargs) of a file read as port type ptype."""
        key = (ptype, path)
        if key not in self.cache:
            read, encode = CONVERTERS[ptype]
            ids, values, implied = read(path)
            if self.order is None:
                self.order, self.first = {c: i for i, c in enumerate(sorted(ids))}, path
            if len(ids) != len(self.order) or len(set(ids)) != len(ids) or any(c not in self.order for c in ids):
                raise ValueError(f"{path}: cell ids differ from {self.first}'s ({len(ids)} vs {len(self.order)} cells)")
            aligned = np.empty_like(values)
            aligned[[self.order[c] for c in ids]] = values  # joined on cell_id, never on row order
            self.cache[key] = (encode(aligned) if encode else aligned, implied)
        return self.cache[key]


def load_catalog(path=CATALOG):
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    ids = [m["id"] for m in doc["metrics"]]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"{path}: duplicate metric ids {dup}")
    return {m["id"]: m for m in doc["metrics"]}, hashlib.sha256(raw).hexdigest()


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


_DATA = None  # per process: the reference (and anything shared) is read once per worker


def _work(task):
    """All metrics of one input combination, in one process. Returns their records."""
    global _DATA
    import sklearn

    sha, group = task
    _DATA = _DATA or Inputs()
    out = []
    for e, f, chosen, subject in group:
        params = inspect.signature(f).parameters
        args, implied = {}, {}
        for p, path in chosen.items():
            args[p], extra = _DATA.get(e["ports"][p], path)
            implied.update({k: v for k, v in extra.items() if k in params})  # a scores port's labels=
        out.append({
            "metric": e["id"],
            "value": float(f(**args, **{**implied, **e.get("kwargs", {})})),
            "subject": subject,
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
        })
    return out


def run(jobs, sha, workers=1):
    """Records of every metric x input combination; combinations run in parallel, one per task.

    A task aligns its own files, so each worker's cell order is its own: fine, since a metric only
    ever sees arrays from one worker. ponytail: per-member parallelism, each worker holds the
    reference (~150 MB at 1M cells); share it via shared memory if workers x RAM becomes the limit.
    """
    groups = {}
    for e, f, paths in jobs:
        ports = list(paths)
        fanned = [p for p in ports if len(paths[p]) > 1]
        for combo in itertools.product(*(paths[p] for p in ports)):
            chosen = dict(zip(ports, combo))
            subject = " | ".join(os.path.dirname(chosen[p]) for p in (fanned or ports))
            groups.setdefault(tuple(sorted(chosen.items())), []).append((e, f, chosen, subject))
    tasks = [(sha, g) for g in groups.values()]
    global _DATA
    _DATA = None  # a fresh cache per run in this process (pool workers start fresh anyway)
    if workers <= 1 or len(tasks) <= 1:
        results = list(map(_work, tasks))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(workers, len(tasks))) as pool:
            results = list(pool.map(_work, tasks))  # in task order: output is deterministic
    return [r for recs in results for r in recs]


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
    p.add_argument("--bind", default="", help="port=input_id,...")
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)),
                   help="processes over input combinations (default: the CPUs this process may use)")
    for legacy in ("--output_dir", "--name"):  # passed by the runtime until api 0.10; unused
        p.add_argument(legacy, help=argparse.SUPPRESS)
    args, rest = p.parse_known_args(argv)

    out = parse_output(args.output)
    catalog, sha = load_catalog()
    jobs = plan(args.metric.split(","), pairs(args.bind), parse_inputs(rest), catalog)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for rec in run(jobs, sha, args.workers):
            fh.write(json.dumps(rec) + "\n")


def every_entry(d, catalog, rng):
    """Every catalog entry, run from files, against its sklearn function called on the arrays.
    Plus: the catalog and its excluded list together cover sklearn.metrics exactly."""
    import sklearn.metrics as skm

    n, classes = 400, np.array(["B cell", "NK", "T cell", "mono", "pDC"])
    ids = [f"c{i:04d}" for i in range(n)]  # sort in index order: index order is the canonical order
    t = rng.integers(0, len(classes), n)
    p = np.where(rng.random(n) < 0.4, rng.integers(0, len(classes), n), t)
    vt = rng.gamma(2.0, 1.0, n) + 0.1  # positive: the deviances and log errors need it
    vp = vt * rng.lognormal(0, 0.3, n)
    prob = rng.dirichlet(np.ones(len(classes)), n)
    prob[np.arange(n), t] += 1.0
    prob /= prob.sum(1, keepdims=True)
    X = rng.normal(size=(len(classes), 4))[t] + rng.normal(scale=0.8, size=(n, 4))

    def write(name, header, cols):
        idx = rng.permutation(n)  # every file lists cells in its own order
        with open(d / name, "w") as f:
            f.write("\t".join(["cell_id", *header]) + "\n")
            f.writelines("\t".join([ids[i], *(str(c[i]) for c in cols)]) + "\n" for i in idx)
        return str(d / name)

    shuffled = rng.permutation(len(classes))  # score columns not in class order: the reader sorts them
    fixtures = {  # port type -> [(path, array as the call should see it)], first = truth, second = prediction
        "partition": [(write("lt.tsv", ["label"], [classes[t]]), classes[t]), (write("lp.tsv", ["label"], [classes[p]]), classes[p])],
        "labels": [(str(d / "lt.tsv"), classes[t]), (str(d / "lp.tsv"), classes[p])],
        "values": [(write("vt.tsv", ["v"], [vt]), vt), (write("vp.tsv", ["v"], [vp]), vp)],
        "scores": [(write("s.tsv", list(classes[shuffled]), [prob[:, j] for j in shuffled]), prob)],
        "embedding": [(write("x.tsv", [f"PC{j + 1}" for j in range(4)], list(X.T)), X)],
    }
    for e in catalog.values():
        used, argv, want = {}, [], {}
        for port, ptype in e["ports"].items():
            path, arr = fixtures[ptype][used.setdefault(ptype, 0)]
            used[ptype] += 1
            argv += [f"--{port}", path]
            want[port] = arr
        main(["--output", f"o={d}/e_{e['id']}.jsonl", "--metric", e["id"], "--workers", "1", *argv])
        got = json.loads(open(d / f"e_{e['id']}.jsonl").readline())["value"]
        f = getattr(skm, e["call"].rpartition(".")[2])
        implied = {"labels": list(classes)} if "scores" in e["ports"].values() and "labels" in inspect.signature(f).parameters else {}
        expect = float(f(**want, **implied, **e.get("kwargs", {})))
        assert np.isclose(got, expect, rtol=1e-9, atol=1e-12), (e["id"], got, expect)
        lo, hi = e["range"]
        assert (lo is None or got >= lo) and (hi is None or got <= hi), (e["id"], got, e["range"])

    # canonical order: the same cells in two file orders give the same arrays (a sampled
    # silhouette would otherwise depend on which file a worker happened to read first)
    other = write("x2.tsv", [f"PC{j + 1}" for j in range(4)], list(X.T))
    a, b = Inputs().get("embedding", fixtures["embedding"][0][0])[0], Inputs().get("embedding", other)[0]
    assert np.array_equal(a, b), "cell order depends on file order"

    doc = yaml.safe_load(CATALOG.read_text())
    have = {fn for fn in skm.__all__ if inspect.isfunction(getattr(skm, fn))}
    placed = {m["call"].rpartition(".")[2] for m in doc["metrics"]} | set(doc["excluded"])
    assert have == placed, f"not placed: {sorted(have - placed)}; stale: {sorted(placed - have)}"
    both = {m["call"].rpartition(".")[2] for m in doc["metrics"]} & set(doc["excluded"])
    assert not both, f"both an entry and excluded: {sorted(both)}"


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
    part = [k for k, e in catalog.items() if list(e["ports"].values()) == ["partition", "partition"]]
    gather = ["--metric", ",".join(part), "--bind", "labels_true=truth,labels_pred=m",  # --m: a prefix of --metric
              "--truth", str(d / "t.tsv"), "--m", *(str(d / f"m{m}" / "p.tsv") for m in range(3))]
    # as the api 0.8 runtime calls it: legacy flags too, --output in its = form
    main([f"--output=metrics_jsonl={d}/x.jsonl", "--output_dir", str(d), "--name", "x", "--workers", "3", *gather])
    main(["--output", f"metrics_jsonl={d}/serial.jsonl", "--workers", "1", *gather])
    assert (d / "x.jsonl").read_text() == (d / "serial.jsonl").read_text(), "parallel and serial runs differ"
    recs = [json.loads(l) for l in open(d / "x.jsonl")]
    assert len(recs) == 3 * len(part), len(recs)
    for r in recs:
        e = catalog[r["metric"]]
        m = int(Path(r["subject"]).name[1:])
        assert r["subject"] == str(d / f"m{m}"), r["subject"]
        want = getattr(skm, e["call"].rpartition(".")[2])(truth, preds[m], **e.get("kwargs", {}))
        assert abs(r["value"] - want) < 1e-12, (r["metric"], m, r["value"], want)
        lo, hi = e["range"]
        assert (lo is None or r["value"] >= lo) and (hi is None or r["value"] <= hi), r

    main(["--output", f"m={d}/one.jsonl", "--metric", "ari",  # no fan-in: subject is every input's dir
          "--labels_true", str(d / "t.tsv"), "--labels_pred", str(d / "m0" / "p.tsv")])
    assert json.loads(open(d / "one.jsonl").readline())["subject"] == f"{d} | {d}/m0"

    every_entry(d, catalog, rng)

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
    fails(lambda: parse_output([]), "exactly one")
    fails(lambda: parse_output(["a=/x", "b=/y"]), "exactly one")
    fails(lambda: parse_output(["/x.jsonl"]), "id=PATH")
    fails(lambda: parse_output(["a=rel.jsonl"]), "absolute")
    with open(d / "short.tsv", "w") as f:
        f.write("cell_id\tcluster\nc0\t1\n")
    inp = Inputs()
    inp.get("partition", str(d / "t.tsv"))
    fails(lambda: inp.get("partition", str(d / "short.tsv")), "cell ids differ")
    print(f"selftest ok: fan-in {len(recs)} records (3 members x {len(part)} partition metrics), all {len(catalog)} catalog entries match sklearn, sklearn.metrics fully placed")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--selftest"] else main()
