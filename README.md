# omni-sklearn-metrics

Omnibenchmark metrics module where `catalog.yaml` is the implementation: each entry names a
function, its typed input ports and its fixed kwargs, and `run.py` resolves it at runtime.
A new sklearn metric is one catalog entry, no code.

Omnibenchmark api >= 0.8.0 only: the output arrives as `--output <id>=<absolute path>` (named
outputs, design 009); `--output_dir`/`--name` are accepted and ignored.

```yaml
# in a benchmark plan (api_version: 0.8.0)
- id: metrics
  gather:
    - {from: clusters_tsv, group_by: data}
  outputs:
    - {id: metrics_jsonl, path: "{module.id}_metrics.jsonl"}
  modules:
    - id: sklearn
      software_environment: sklearn
      repository: {url: …/omni-sklearn-metrics, commit: <sha>, entrypoint: run}
      parameters:
        - metric: "ari,nmi"
          bind: "labels_true=reference_tsv,labels_pred=clusters_tsv"  # reference_tsv: however the plan provides it
```

- Fan-in: a port bound to several paths (a `gather`) runs each metric once per path; every file
  is read, joined on `cell_id` and int-coded once, so the rest is one sklearn call per pair.
  Combinations run in parallel (`--workers`, default: the CPUs the process may use).
  Subject = the fanned-out file's directory (else every input's directory).
- Port types: `partition` (TSV: `cell_id` + one label column). Inputs are joined on `cell_id`;
  differing cell sets are an error, never a silent drop.
- `call` must start with `sklearn.metrics.` (`ALLOWED` in run.py): the catalog is code.
- Output: the `--output` path, one jsonl record per metric and combination, with the catalog fields, the catalog's
  sha256 and the sklearn version in `attrs`.
- Check: `pixi run selftest` (every catalog metric, fanned over 3 members, against sklearn called directly).
- Env: `envs/sklearn.yml`, exported from pixi.toml by `pixi run export-env`.

The resolver and converters are meant to move into obkit once a second module uses them.
