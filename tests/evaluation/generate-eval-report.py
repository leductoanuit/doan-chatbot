"""Generate a markdown evaluation report from a JSON eval result file.

Usage:
  python tests/evaluation/generate-eval-report.py results/eval-summary-*.json
  python tests/evaluation/generate-eval-report.py  # auto-picks latest report
"""

import json
import sys
from pathlib import Path

from eval_markdown_renderer import generate_markdown

RESULTS_DIR = Path(__file__).parent / "results"


def load_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) > 1:
        report_path = Path(sys.argv[1])
    else:
        if not RESULTS_DIR.exists():
            print(f"Results directory not found: {RESULTS_DIR}")
            print("Run: python tests/evaluation/run-evaluation.py first")
            sys.exit(1)
        reports = sorted(RESULTS_DIR.glob("eval-summary-*.json"))
        if not reports:
            print("No eval-summary-*.json found in", RESULTS_DIR)
            sys.exit(1)
        report_path = reports[-1]
        print(f"Using latest report: {report_path}")

    report = load_report(report_path)
    md = generate_markdown(report)

    out_path = report_path.with_suffix(".md")
    out_path.write_text(md, encoding="utf-8")
    print(f"Markdown report saved to: {out_path}")
    print("\n" + md[:500] + "\n...")


if __name__ == "__main__":
    main()
