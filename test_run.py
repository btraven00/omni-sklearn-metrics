#!/usr/bin/env python3
"""Tests for run.py: python test_run.py (pixi run test). Plain asserts, one function per check."""

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import sklearn.metrics as skm
import yaml

import run

RNG = np.random.default_rng(0)
CLASSES = np.array(["B cell", "NK", "T cell", "mono", "pDC"])


# --- helpers -----------------------------------------------------------------------------------

def tmpdir():
    return Path(tempfile.mkdtemp())


def write_tsv(path, header, columns, n=None):
    """A cell_id TSV, cells c0000.. listed in a random order (inputs never rely on row order)."""
    n = n or len(columns[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(["cell_id", *header]) + "\n")
        for i in RNG.permutation(n):
            f.write("\t".join([f"c{i:04d}", *(str(c[i]) for c in columns)]) + "\n")
    return str(path)


def run_cli(out, *argv):
    """main() with --workers 1; the records it wrote."""
    run.main(["--output", f"metrics={out}", "--workers", "1", *argv])
    return [json.loads(line) for line in open(out)]


def fails(fn, *fragments):
    """fn raises ValueError whose message contains every fragment."""
    try:
        fn()
    except ValueError as e:
        for frag in fragments:
            assert frag in str(e), f"{frag!r} not in error: {e}"
    else:
        raise AssertionError(f"expected ValueError mentioning {fragments}")


def sklearn_fn(entry):
    return getattr(skm, entry["call"].rpartition(".")[2])


# --- every catalog entry against sklearn -------------------------------------------------------

def fixtures(d, n=400):
    """data type -> [(path, array as sklearn should see it)]; first = truth, second = prediction."""
    t = RNG.integers(0, len(CLASSES), n)
    p = np.where(RNG.random(n) < 0.4, RNG.integers(0, len(CLASSES), n), t)
    vt = RNG.gamma(2.0, 1.0, n) + 0.1  # positive: every regression entry's domain
    vp = vt * RNG.lognormal(0, 0.3, n)
    prob = RNG.dirichlet(np.ones(len(CLASSES)), n)
    prob[np.arange(n), t] += 1.0
    prob /= prob.sum(1, keepdims=True)
    X = RNG.normal(size=(len(CLASSES), 4))[t] + RNG.normal(scale=0.8, size=(n, 4))
    shuffled = RNG.permutation(len(CLASSES))  # score columns out of class order: the reader sorts them
    labels = [(write_tsv(d / "lt.tsv", ["label"], [CLASSES[t]]), CLASSES[t]),
              (write_tsv(d / "lp.tsv", ["label"], [CLASSES[p]]), CLASSES[p])]
    return {
        "partition": labels,
        "labels": labels,
        "values": [(write_tsv(d / "vt.tsv", ["v"], [vt]), vt), (write_tsv(d / "vp.tsv", ["v"], [vp]), vp)],
        "scores": [(write_tsv(d / "s.tsv", list(CLASSES[shuffled]), [prob[:, j] for j in shuffled]), prob)],
        "embedding": [(write_tsv(d / "x.tsv", ["PC1", "PC2", "PC3", "PC4"], list(X.T)), X)],
    }


def test_every_entry_matches_sklearn():
    d = tmpdir()
    fx = fixtures(d)
    catalog = run.load_catalog()
    for e in catalog.metrics.values():
        used, argv, arrays = {}, [], {}
        for arg, dtype in e["args"].items():
            path, array = fx[dtype][used.get(dtype, 0)]
            used[dtype] = used.get(dtype, 0) + 1
            argv += [f"--{arg}", path]
            arrays[arg] = array
        got = run_cli(d / f"{e['id']}.jsonl", "--metric", e["id"], *argv)[0]["value"]
        f = sklearn_fn(e)
        implied = {"labels": list(CLASSES)} if "scores" in e["args"].values() and "labels" in inspect.signature(f).parameters else {}
        expect = float(f(**arrays, **implied, **e.get("kwargs", {})))
        assert np.isclose(got, expect, rtol=1e-9, atol=1e-12), (e["id"], got, expect)


def test_catalog_covers_sklearn_metrics():
    doc = yaml.safe_load(run.CATALOG.read_text())
    have = {fn for fn in skm.__all__ if inspect.isfunction(getattr(skm, fn))}
    entries = {m["call"].rpartition(".")[2] for m in doc["metrics"]}
    excluded = set(doc["excluded"])
    assert have == entries | excluded, f"not placed: {sorted(have - entries - excluded)}; stale: {sorted(entries | excluded - have)}"
    assert not entries & excluded, f"both an entry and excluded: {sorted(entries & excluded)}"


# --- fan-in, order, CLI -----------------------------------------------------------------------

def test_fan_in_parallel_equals_serial():
    d = tmpdir()
    n, truth = 500, RNG.integers(0, 5, 500)
    preds = []
    for m in range(3):  # three members of a gather, agreeing with the truth to different degrees
        lab = RNG.integers(0, 7, n)
        lab[: 100 * (m + 1)] = truth[: 100 * (m + 1)]
        preds.append(lab)
        write_tsv(d / f"m{m}" / "p.tsv", ["cluster"], [lab])
    write_tsv(d / "t.tsv", ["cluster"], [truth])
    catalog = run.load_catalog()
    part = [k for k, e in catalog.metrics.items() if list(e["args"].values()) == ["partition", "partition"]]
    gather = ["--metric", ",".join(part), "--bind", "labels_true=truth,labels_pred=m",  # --m: a prefix of --metric
              "--truth", str(d / "t.tsv"), "--m", *(str(d / f"m{m}" / "p.tsv") for m in range(3))]
    # as the api 0.8 runtime calls it: legacy flags too, --output in its = form
    run.main([f"--output=metrics={d}/par.jsonl", "--output_dir", str(d), "--name", "x", "--workers", "3", *gather])
    run.main(["--output", f"metrics={d}/ser.jsonl", "--workers", "1", *gather])
    assert (d / "par.jsonl").read_text() == (d / "ser.jsonl").read_text(), "parallel and serial runs differ"
    recs = [json.loads(line) for line in open(d / "par.jsonl")]
    assert len(recs) == 3 * len(part)
    for r in recs:
        m = int(Path(r["subject"]).name[1:])
        assert r["subject"] == str(d / f"m{m}")
        e = catalog.metrics[r["metric"]]
        assert np.isclose(r["value"], sklearn_fn(e)(truth, preds[m], **e.get("kwargs", {})), rtol=1e-12)


def test_single_pair_subject_is_every_input_dir():
    d = tmpdir()
    a = write_tsv(d / "a" / "x.tsv", ["c"], [RNG.integers(0, 3, 50)])
    b = write_tsv(d / "b" / "x.tsv", ["c"], [RNG.integers(0, 3, 50)])
    rec = run_cli(d / "o.jsonl", "--metric", "ari", "--labels_true", a, "--labels_pred", b)[0]
    assert rec["subject"] == f"{d}/a | {d}/b"


def test_cell_order_does_not_depend_on_file_order():
    d = tmpdir()
    X = RNG.normal(size=(100, 3))
    a = write_tsv(d / "a.tsv", ["x", "y", "z"], list(X.T))  # two random row orders of the same cells
    b = write_tsv(d / "b.tsv", ["x", "y", "z"], list(X.T))
    types = run.load_catalog().types
    assert np.array_equal(run.Inputs(types).get("embedding", a)[0], run.Inputs(types).get("embedding", b)[0])


def test_cli_errors():
    catalog = run.load_catalog()
    fails(lambda: run.parse_output([]), "exactly one")
    fails(lambda: run.parse_output(["a=/x", "b=/y"]), "exactly one")
    fails(lambda: run.parse_output(["/x.jsonl"]), "id=PATH")
    fails(lambda: run.parse_output(["a=rel.jsonl"]), "absolute")
    fails(lambda: run.parse_inputs(["--a"]), "without a path")
    fails(lambda: run.plan(["nope"], {}, {}, catalog), "unknown metric ids", "nope")
    fails(lambda: run.plan(["ari"], {}, {}, catalog), "argument labels_true", "not given")
    fails(lambda: run.resolve({"id": "evil", "call": "os.system", "args": {}}), "outside")
    fails(lambda: run.resolve({"id": "bad", "call": "sklearn.metrics.adjusted_rand_score", "args": {"x": "partition"}}), "no argument 'x'")


# --- 1. catalog consistency -------------------------------------------------------------------

def test_inconsistent_catalog_lists_every_problem():
    doc = yaml.safe_load(run.CATALOG.read_text())
    doc["types"]["values"]["rules"].append("shiny")
    doc["types"]["embedding"]["columns"] = "many"
    doc["metrics"].append(dict(doc["metrics"][0]))  # duplicate id
    doc["metrics"][1] = {**doc["metrics"][1], "args": {"labels_true": "nonsense", "labels_pred": "partition"},
                         "requires": {"y": ["finite"], "labels_pred": ["glossy"]}}
    path = tmpdir() / "catalog.yaml"
    path.write_text(yaml.safe_dump(doc))
    fails(lambda: run.load_catalog(path), "duplicate metric id 'ari'", "unknown rule 'shiny'", "columns must be",
          "unknown data type 'nonsense'", "requires names 'y'", "unknown rule 'glossy'")


# --- 2. pre-flight: headers, before any data is read ------------------------------------------

def test_preflight_checks_headers_before_reading_data():
    d = tmpdir()
    catalog = run.load_catalog()
    ok = write_tsv(d / "ok.tsv", ["v"], [RNG.random(20)])
    garbage = d / "garbage.tsv"
    garbage.write_text("cell_id\tv\nc0000\tnot-a-number\n")  # a good header over bad data
    two_cols = write_tsv(d / "two.tsv", ["v", "w"], [RNG.random(20), RNG.random(20)])
    no_id = d / "noid.tsv"
    no_id.write_text("cell\tv\nc0000\t1\n")
    two_classes = write_tsv(d / "s2.tsv", ["a", "b"], [np.full(20, 0.5), np.full(20, 0.5)])

    def plan(**inputs):
        return lambda: run.plan(list(inputs.pop("metric")), {}, {k: [str(v)] for k, v in inputs.items()}, catalog)

    # the bad header is reported although the other file, read first in a run, has bad data
    fails(plan(metric=["r2"], y_true=garbage, y_pred=two_cols), str(two_cols), "needs 1 column(s)")
    fails(plan(metric=["r2"], y_true=ok, y_pred=no_id), "first column must be cell_id")
    fails(plan(metric=["r2"], y_true=ok, y_pred=d / "missing.tsv"), "no such file")
    fails(plan(metric=["log_loss"], y_true=ok, y_pred=two_classes), "(scores)", "needs >=3 column(s)")


# --- 3. data type rules, once per file -----------------------------------------------------------

def test_type_rules_name_the_file_and_rule():
    d = tmpdir()
    n = 30
    good = write_tsv(d / "good.tsv", ["v"], [RNG.random(n) + 1])
    nan = write_tsv(d / "nan.tsv", ["v"], [np.where(np.arange(n) == 3, np.nan, 1.0)])
    text = d / "text.tsv"
    text.write_text("cell_id\tv\n" + "".join(f"c{i:04d}\t{'x' if i == 0 else 1}\n" for i in range(n)))
    labels = write_tsv(d / "l.tsv", ["label"], [CLASSES[np.arange(n) % 3]])
    prob = np.full((n, 3), 1 / 3)
    unnormalized = write_tsv(d / "s_sum.tsv", list(CLASSES[:3]), [prob[:, 0] * 2, prob[:, 1], prob[:, 2]])
    negative = prob.copy()
    negative[0] = [1.2, -0.2, 0.0]
    out_of_unit = write_tsv(d / "s_unit.tsv", list(CLASSES[:3]), list(negative.T))

    o = d / "o.jsonl"
    fails(lambda: run_cli(o, "--metric", "r2", "--y_true", good, "--y_pred", nan), "nan.tsv (values)", "finite: 1 value(s) NaN or inf")
    fails(lambda: run_cli(o, "--metric", "r2", "--y_true", good, "--y_pred", str(text)), "text.tsv", "not a numeric table")
    fails(lambda: run_cli(o, "--metric", "log_loss", "--y_true", labels, "--y_pred", unnormalized), "probabilities", "rows not summing to 1")
    fails(lambda: run_cli(o, "--metric", "log_loss", "--y_true", labels, "--y_pred", out_of_unit), "probabilities", "outside [0, 1]")


# --- 4. per-metric requirements ----------------------------------------------------------------

def test_requires_names_metric_argument_and_rule():
    d = tmpdir()
    n = 30
    pos = write_tsv(d / "pos.tsv", ["v"], [RNG.random(n) + 1])
    zero = write_tsv(d / "zero.tsv", ["v"], [np.where(np.arange(n) == 0, 0.0, 1.0)])
    minus_one = write_tsv(d / "m1.tsv", ["v"], [np.where(np.arange(n) == 0, -1.0, 1.0)])
    one_cluster = write_tsv(d / "one.tsv", ["c"], [np.zeros(n, int)])
    X = write_tsv(d / "x.tsv", ["a", "b"], [RNG.random(n), RNG.random(n)])
    o = d / "o.jsonl"
    fails(lambda: run_cli(o, "--metric", "mean_gamma_deviance", "--y_true", pos, "--y_pred", zero),
          "mean_gamma_deviance: argument y_pred", "positive: 1 value(s) <= 0")
    fails(lambda: run_cli(o, "--metric", "mean_squared_log_error", "--y_true", minus_one, "--y_pred", pos),
          "argument y_true", "above_minus_one")
    fails(lambda: run_cli(o, "--metric", "silhouette", "--X", X, "--labels", one_cluster),
          "silhouette: argument labels", "several_groups: 1 distinct labels")
    # Poisson allows zeros in y_true: the same file passes there
    assert run_cli(o, "--metric", "mean_poisson_deviance", "--y_true", zero, "--y_pred", pos)[0]["value"] >= 0


def test_scores_must_have_a_column_for_every_label():
    d = tmpdir()
    n = 30
    labels = write_tsv(d / "l.tsv", ["label"], [CLASSES[np.arange(n) % 4]])  # 4 classes
    scores = write_tsv(d / "s.tsv", list(CLASSES[:3]), [np.full(n, 1 / 3)] * 3)  # 3 columns
    fails(lambda: run_cli(d / "o.jsonl", "--metric", "log_loss", "--y_true", labels, "--y_pred", scores),
          "argument y_true has labels ['mono']", "no column in the scores file")


# --- 5. results ---------------------------------------------------------------------------------

def test_result_checks():
    entry = {"id": "m", "range": [0, 1]}
    assert run.check_result(entry, 1 + 1e-12) == (1 + 1e-12, None)  # inside the float slack
    assert run.check_result(entry, float("nan")) == (None, "sklearn returned nan")
    fails(lambda: run.check_result(entry, 1.5), "m: value 1.5 outside its declared range [0, 1]")
    fails(lambda: run.check_result({"id": "m", "range": [None, 1]}, 2.0), "outside")
    assert run.check_result({"id": "m", "range": [0, None]}, 1e9) == (1e9, None)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
    sys.exit(0)
