import os
import sys
import argparse
project_root = os.path.abspath("/ceph/harris/hypnose/hypnose-analysis")
if project_root not in sys.path:
    sys.path.append(project_root)
from classification_utils import batch_analyze_sessions
from datetime import datetime, timedelta

def generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    delta = timedelta(days=1)
    dates = []
    while start <= end:
        dates.append(start.strftime("%Y%m%d"))
        start += delta
    return dates


def generate_dates_from(start_date, max_days=365):
    start = datetime.strptime(start_date, "%Y%m%d")
    delta = timedelta(days=1)
    dates = []
    for _ in range(max_days):
        dates.append(start.strftime("%Y%m%d"))
        start += delta
    return dates

def main():
    parser = argparse.ArgumentParser(description="Run batch analysis for sessions on HPC.")
    parser.add_argument("--subjids", nargs="*", type=int, default=None, help="List of subject IDs (or None).")
    parser.add_argument("--dates", nargs="*", type=int, help="List of specific dates to analyze.")
    parser.add_argument("--start_date", type=str, help="Start date for analysis (YYYYMMDD).")
    parser.add_argument("--end_date", type=str, help="End date for analysis (YYYYMMDD).")
    parser.add_argument("--save", action="store_true", help="Save the results.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--print_summary", action="store_true", help="Print summary of results.")

    args = parser.parse_args()

    if args.start_date and args.end_date:
        args.dates = generate_date_range(args.start_date, args.end_date)

    if args.start_date and not args.end_date:
        args.dates = generate_dates_from(args.start_date)

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