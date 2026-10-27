"""
__main__.py

CLI: `python -m locus.experiments run --config <path>`.

Runs one experiment end-to-end, prints the outcome, and verifies
reproducibility if requested (P0 exit criterion: diff < 1e-9).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from locus.experiments.runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m locus.experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run an experiment end-to-end")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--registry", type=Path, default=Path("trials/registry.jsonl"))
    run.add_argument("--reports-dir", type=Path, default=Path("reports"))
    run.add_argument("--data-root", type=Path, default=Path("data"))
    run.add_argument("--end-date", default="2026-10-01")
    run.add_argument(
        "--no-register",
        action="store_true",
        help="Do not append to registry (reproduction mode)",
    )
    run.add_argument(
        "--verify-reproducible",
        action="store_true",
        help="Run twice and report max absolute diff on per-period returns",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        result = run_experiment(
            config_path=args.config,
            registry_path=args.registry,
            reports_dir=args.reports_dir,
            data_root=args.data_root,
            end_date=args.end_date,
            register=not args.no_register,
            verify_reproducible=args.verify_reproducible,
        )
        print(f"experiment:  {result.config_id}")
        print(f"sharpe:      {result.validation.sharpe:+.6f}")
        print(f"baseline:    {result.baseline_sharpe:+.6f}")
        print(f"verdict:     {result.validation.verdict}")
        print(f"n_trials:    {result.n_trials_used}")
        print(f"report:      {result.report_path}")
        if result.reproducibility_diff is not None:
            print(f"repro diff:  {result.reproducibility_diff:.3e}")
            if result.reproducibility_diff < 1e-9:
                print("PASS: diff < 1e-9 (P0 exit criterion met)")
            else:
                print("FAIL: diff >= 1e-9")
                raise SystemExit(1)


if __name__ == "__main__":
    main()
