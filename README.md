# omni-sklearn-metrics

Omnibenchmark declarative metrics module, based on skilearn: each entry in the `catalog.yaml` 
declares a sklearn function, the data type of each of its arguments, its fixed
kwargs, its direction and its range. The entrypoint in `run.py` resolves it at runtime.
A new sklearn metric is one catalog entry, no new code needed.

Supports omnibenchmark api >= 0.8.0 only: the output arrives as `--output
<id>=<absolute path>` (named outputs, design 009); `--output_dir`/`--name` are
accepted and ignored.

```yaml
# in a benchmark plan (api_version: 0.8.0)
- id: metrics
  gather:
    - {from: clusters_tsv, group_by: data}
  outputs:
    - {id: metrics_jsonl, path: "{module.id}_metrics.jsonl"}
  modules:
    - id: metrics-sklearn
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

1. **catalog**: every entry's function, arguments, kwargs, data types and rules exist.
2. **pre-flight**, before any data is read: every requested id and binding, and every bound
   file's header (exists, `cell_id` first, column count of its data type).
3. **inputs**, once per file: cell ids identical across files and unique; the data type's
   rules (`finite`; `probabilities`: in [0, 1], rows summing to 1).
4. **metric**: the entry's `requires` (sklearn's domains, e.g. `positive` for the gamma
   deviance, `several_groups` for silhouette); a scores file has a column for every label.
5. **result**: a value outside the entry's declared range fails the run; NaN is written as
   `null` with a note.

## Running

- Fan-in: an argument bound to several paths (a `gather`) runs each metric once per path.
  Combinations run in parallel (`--workers`, default: the CPUs the process may use); each file
  is read, joined on `cell_id` and checked once per worker. Cells are put in sorted `cell_id`
  order, so results do not depend on the order files list them in.
- Output: the `--output` path, one jsonl record per metric and combination (sc-brrr
  `specs/README.md`): subject = the fanned-out input's directory (else every input's directory),
  with the catalog fields, the catalog's sha256 and the sklearn version in `attrs`.
- `call` must start with `sklearn.metrics.` (`ALLOWED` in run.py): the catalog is code.
- Tests: `pixi run test` (test_run.py: every entry against sklearn, every check, fan-in).
- Env: `envs/sklearn.yml`, exported from pixi.toml by `pixi run export-env`.

The resolver, readers and checks are meant to move into obkit once a second module uses them.
