# scripts

Terminal entry points for running the pipeline. They are thin wrappers over
functions in `src/hypnose/` and contain no analysis logic. Run from the repo
root in the project conda environment; the scripts add `src/` to the path, so no
install is required.

| Script | What it does |
| --- | --- |
| `run_trial_classification.py` | Trial classification → writes `trial_data` (+ summary) to derivatives |
| `run_metrics_analysis.py` | Behavioural metric analysis (reads saved classification results) |
| `batch_process.py` | Runs trial classification, then metric analysis |

Run trial classification **before** metric analysis — metrics read the saved
classification results from the derivatives tree.

## Arguments (shared)

| Argument | Meaning |
| --- | --- |
| `--subjids ID [ID ...]` | One or more subject ids. Omit to run **all** subjects found in rawdata. |
| `--dates D [D ...]` | One or more specific dates `YYYYMMDD`. |
| `--date-range START END` | Inclusive date range `YYYYMMDD YYYYMMDD` (alternative to `--dates`). |

`--dates` and `--date-range` are mutually exclusive; omit both to run **all dates**
for the selected subjects. Subjects/dates with no data are validated and skipped
with a clear message (`hypnose.qc.validate.validate_subject`).

Script-specific:

| Argument | Script(s) | Meaning |
| --- | --- | --- |
| `--no-save` | all | Do not write derivatives / metrics output |
| `--no-summary` | run_trial_classification | Suppress the merged per-session summary |
| `--verbose` | run_trial_classification, batch_process | Verbose per-run logging |
| `--quiet` | run_metrics_analysis | Suppress per-session logging |
| `--protocol STR` | run_metrics_analysis, batch_process | Only sessions whose stage name contains `STR` |

## Examples

```bash
# one subject, one date
python scripts/run_trial_classification.py --subjids 53 --dates 20260528

# one subject, several specific dates
python scripts/run_trial_classification.py --subjids 53 --dates 20260520 20260528

# multiple subjects, a date range
python scripts/run_trial_classification.py --subjids 53 58 --date-range 20260501 20260531

# all subjects, all dates
python scripts/run_trial_classification.py

# metrics for a protocol only (run after classification)
python scripts/run_metrics_analysis.py --subjids 53 --date-range 20260501 20260531 --protocol singrew

# classification + metrics in one go
python scripts/batch_process.py --subjids 53 58 --dates 20260528
```
