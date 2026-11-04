# omni-sklearn-metrics

Omnibenchmark metrics module where `catalog.yaml` is the implementation: each entry names a
function, its typed input ports and its fixed kwargs, and `run.py` resolves it at runtime.
A new sklearn metric is one catalog entry, no code.

```yaml
# in a benchmark plan
- id: sklearn
  repository: {url: …/omni-sklearn-metrics, commit: <sha>, entrypoint: run}
  parameters:
    - metric: "ari,nmi"
      bind: "labels_true=reference_tsv,labels_pred=clusters_tsv"
```

- Port types: `partition` (TSV: `cell_id` + one label column). Inputs are joined on `cell_id`;
  differing cell sets are an error, never a silent drop.
- `call` must start with `sklearn.metrics.` (`ALLOWED` in run.py): the catalog is code.
- Output: `{name}_metrics.jsonl`, one record per metric, with the catalog fields, the catalog's
  sha256 and the sklearn version in `attrs`.
- Check: `pixi run selftest` (every catalog metric against sklearn called directly).

The resolver and converters are meant to move into obkit once a second module uses them.
