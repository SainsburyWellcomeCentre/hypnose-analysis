# scripts

Terminal entry points for running the pipeline (populated during the restructuring):

- `run_trial_classification.py` — trial classification for given subjid(s)/date(s)
- `run_metrics_analysis.py` — metric analysis for given subjid(s)/date(s)
- `batch_process.py` — run both for given subjid(s)/date(s)

These are thin wrappers over functions in `src/hypnose/`; they contain no analysis logic.
