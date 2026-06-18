# Restructuring regression harness

Golden-master safety net for the `hypnose-restructuring` effort. The restructuring
is done as a series of **pure moves** (split / rename / relocate, no logic changes),
so the pipeline output must stay **byte-for-byte identical** at every step. This
harness proves it.

## What it checks

For each session in [`sessions.yml`](sessions.yml) it fingerprints the two outputs
that must not change:

- **`trial_data`** (trial classification) — md5 of the canonical CSV (sorted
  columns, no index; not parquet bytes, not the manifest/summary).
- **metrics** — md5 of the metrics dict returned by `run_all_metrics` (timestamp-free).

It reads the real, read-only `rawdata` and redirects all derivatives I/O to a
throwaway temp dir (`HYPNOSE_DATA_ROOT` is untouched; `HYPNOSE_DERIVATIVES_ROOT`
is pointed at the temp dir per session), so it never writes to the server.

## Sessions (chosen for code-path coverage)

single-reward · standard doubles · odourdiscrimination special-case · multi-run
merge · hidden-rule (`location0123`) · pred-seq triples.

## Usage

Run in the pinned conda env recorded in `fixtures/env.json`.

```bash
# (baseline only, already done on main) write the ground-truth fixtures:
python src/hypnose/qc/regression.py --generate

# after EVERY restructuring step:
python src/hypnose/qc/regression.py        # exit 0 = GREEN, 1 = RED
```

A RED result means a "move" changed output and was not pure — revert/fix before
continuing. If you ever *intend* to change output (e.g. the deferred logic
consolidation), regenerate fixtures deliberately in a separate, reviewed commit.

## Notes

- All pipeline imports live in one block in [`_common.py`](_common.py); update only
  those lines as modules move during the restructuring — the md5s must not change.
- Determinism was verified by running the comparison twice on the baseline (both GREEN).
