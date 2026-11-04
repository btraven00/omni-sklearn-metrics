# omni-sklearn-metrics

Omnibenchmark declarative metrics module, based on sklearn: each entry in the `catalog.yaml`
declares a sklearn function, the data type of each of its arguments, its fixed
kwargs, its direction and its range. The entrypoint in `run.py` resolves it at runtime.
A new sklearn metric is one catalog entry, no new code needed.

Supports omnibenchmark api >= 0.8.0 only: the output arrives as `--output
<id>=<absolute path>` (named outputs, design 009); `--output_dir`/`--name` are
accepted and ignored.

**Known issue:** omnibenchmark main passes `--output` to ordinary stages but not yet to
`gather` stages, so under a gather the module stops with "expected exactly one --output"
until that lands upstream.

```yaml
# in a benchmark plan (api_version: 0.8.0)
- id: metrics
  gather:
    - {from: clusters_tsv, group_by: data}
  outputs:
    - {id: metrics_jsonl, path: "{module.id}_metrics.jsonl"}
  modules:
    - id: metrics_sklearn   # api 0.8: ids are letters, digits and underscores
      software_environment: sklearn
      repository: {url: …/omni-sklearn-metrics, commit: <sha>, entrypoint: run}
      parameters:
        - metric: "ari,nmi"
          # function argument = plan input id; reference_tsv: however the plan provides it
          bind: "labels_true=reference_tsv,labels_pred=clusters_tsv"
```

## Catalog

54 entries over every scalar metric in `sklearn.metrics` (clustering agreement, cluster
separation, classification, class probabilities, regression); every other function is under
`excluded:` with the reason. A test fails if the two lists stop covering `sklearn.metrics`
exactly, e.g. after a sklearn upgrade.

Data types (`types:` in catalog.yaml), all TSVs with `cell_id` first: `partition` and `labels`
(one label column; int-coded vs kept), `values` (one numeric column), `scores` (one probability
column per class, >= 3, header = class names, passed as `labels=`), `embedding` (numeric
columns). Omnibenchmark has inputs and outputs by id but no typed ports yet; these types are
what such ports could be typed with.

## Checks

Every error names the metric, the argument and the file. In the order they run:

1. **catalog**: every entry's function, arguments, kwargs, data types, rules, direction and
   range exist and are well-formed.
2. **pre-flight**, before any data is read: requested ids (each once), `--bind` (every key an
   argument of a requested metric, no input named like a module flag), pairing, and every bound
   file's header (exists, `cell_id` first, column count of its data type, no duplicate columns).
3. **inputs**, once per file: every row as wide as the header (no blank lines or missing
   values); cell ids unique and identical across files; the data type's rules (`finite`;
   `probabilities`: in [0, 1], rows summing to 1 within sklearn's tolerance).
4. **metric**: the entry's `requires` (sklearn's domains, e.g. `positive` for the gamma
   deviance, `several_groups` for silhouette); a scores file has a column for every label, and
   for functions that infer the classes from the labels (`average_precision`) every column a label.
5. **result**: a value outside the entry's declared range fails the run; NaN is written as
   `null` with a note.

## Running

- Fan-in: an argument bound to several paths (a `gather`) runs each metric once per path.
  When several arguments get several files, `--pairing lineage` (default) pairs the files of
  one pipeline, whatever order the gather lists them in: omnibenchmark nests a node's directory
  under its deepest input's (design 007 §3.1.1), so an embedding pairs with the clusters whose
  directory is the same or below it. One embedding may serve several clusterings; a prediction
  file matching no or several embeddings, or an embedding matching nothing, is an error.
  `--pairing product` pairs every file with every other but never with itself
  (replicate-to-replicate agreement).
- Combinations run in parallel: `--workers`, default the CPU budget (affinity, capped by a
  cgroup quota) and at most 8, one BLAS thread each. Each file is read, joined on `cell_id` and
  checked once per worker; a member's files are dropped once it is done, files shared by
  several combinations (the reference) are kept. Cells are put in sorted `cell_id` order, so
  results do not depend on the order files list them in.
- Output: the `--output` path, one jsonl record per metric and combination (sc-brrr
  `specs/README.md`), with the catalog fields, the catalog's sha256 and the sklearn version in
  `attrs`. The subject is the directory of the file fed to the entry's last argument (the
  prediction), relative to the omnibenchmark output root (the directory holding `.metadata/`),
  or absolute outside one. Metric ids are this catalog's, not sc-brrr's `specs/metrics.yaml`.
- `call` must start with `sklearn.metrics.` (`ALLOWED` in run.py): the catalog is code.
- Tests: `pixi run test` (test_run.py: every entry against sklearn, every check, fan-in).
- Env: `envs/sklearn.yml`, exported from pixi.toml by `pixi run export-env`.

The resolver, readers and checks are meant to move into obkit once a second module uses them.
