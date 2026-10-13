from __future__ import annotations

import argparse
import json
from pathlib import Path


def match(pred: list[float], truth: list[float], tol: float) -> int:
    used = [False] * len(truth)
    hits = 0
    for p in sorted(pred):
        best, best_d = -1, tol
        for i, t in enumerate(truth):
            if not used[i] and abs(p - t) <= best_d:
                best, best_d = i, abs(p - t)
        if best >= 0:
            used[best] = True
            hits += 1
    return hits


def evaluate(gt: dict[str, object], report: dict[str, object], tol: float) -> dict[str, object]:
    count, truth_count = int(report["count"]), int(gt["count"])  # type: ignore[call-overload]
    out: dict[str, object] = {"count": count, "truth": truth_count, "count_error": count - truth_count,
                              "count_error_pct": round(100 * (count - truth_count) / truth_count, 1)
                              if truth_count else None}
    if "crossings_s" in gt:
        truth = [float(t) for t in gt["crossings_s"]]  # type: ignore[attr-defined]
        pred = [float(e["time_s"]) for e in report["events"]]  # type: ignore[attr-defined, index]
        hits = match(pred, truth, tol)
        out["precision"] = round(hits / len(pred), 3) if pred else None
        out["recall"] = round(hits / len(truth), 3) if truth else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=0.5, help="seconds")
    a = p.parse_args()
    res = evaluate(json.loads(a.gt.read_text()), json.loads(a.report.read_text()), a.tolerance)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
