import os
import sys
import argparse
project_root = os.path.abspath("/ceph/harris/hypnose/hypnose-analysis")
if project_root not in sys.path:
    sys.path.append(project_root)
from classification_utils import batch_analyze_sessions

def main():
    parser = argparse.ArgumentParser(description="Run batch analysis for sessions on HPC.")
    parser.add_argument("--subjids", nargs="*", type=int, default=None, help="List of subject IDs (or None).")
    parser.add_argument("--dates", nargs="*", type=int, required=True, help="List of dates to analyze.")
    parser.add_argument("--save", action="store_true", help="Save the results.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--print_summary", action="store_true", help="Print summary of results.")
    parser.add_argument("--max_runs", type=int, default=32, help="Maximum number of runs to analyze per session.")

    args = parser.parse_args()

    # Run the batch analysis
    batch_analyze_sessions(
        subjids=args.subjids,
        dates=args.dates,
        save=args.save,
        verbose=args.verbose,
        print_summary=args.print_summary,
        max_runs=args.max_runs,
    )

if __name__ == "__main__":
    main()