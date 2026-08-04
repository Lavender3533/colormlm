#!/usr/bin/env python3
"""以三个独立进程运行多随机种子盲复合坐标门。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from axis_core import read_json, write_json


# 固定覆盖四个不同二项与三个不同三项隐藏坐标；不是运行后挑选通过样本。
SEEDS = (17, 28, 43, 52, 71, 88, 101)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, text=True, encoding="utf-8", capture_output=True)


def main() -> None:
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    runs_dir = root / "runs"
    evaluations = []
    for seed in SEEDS:
        run_dir = runs_dir / f"seed_{seed}"
        discovery = run_dir / "discovery.json"
        holdout = run_dir / "sealed_holdout.json"
        manifest = run_dir / "manifest.json"
        frame = run_dir / "frame.json"
        evaluation = run_dir / "evaluation.json"
        _run(
            [
                sys.executable,
                str(root / "generate_world.py"),
                "--output-dir",
                str(run_dir),
                "--seed",
                str(seed),
            ]
        )
        _run(
            [
                sys.executable,
                str(root / "synthesize_frame.py"),
                "--discovery",
                str(discovery),
                "--output",
                str(frame),
            ]
        )
        _run(
            [
                sys.executable,
                str(root / "evaluate_frame.py"),
                "--discovery",
                str(discovery),
                "--frame",
                str(frame),
                "--manifest",
                str(manifest),
                "--holdout",
                str(holdout),
                "--output",
                str(evaluation),
            ]
        )
        receipt = read_json(evaluation)
        evaluations.append(
            {
                "seed": seed,
                "passed": receipt["passed"],
                "frame": receipt["frame"],
                "metrics": receipt["metrics"],
                "assertions": receipt["assertions"],
                "provenance": receipt["provenance"],
                "hidden_axis_after_scoring": receipt["private_audit_revealed_after_scoring"],
            }
        )

    hidden_canonicals = [
        item["hidden_axis_after_scoring"]["hidden_axis"]["canonical"]
        for item in evaluations
    ]
    coverage = {
        "two_term_worlds": sum(len(item["frame"]["terms"]) == 2 for item in evaluations),
        "three_term_worlds": sum(len(item["frame"]["terms"]) == 3 for item in evaluations),
        "unique_hidden_axes": len(set(hidden_canonicals)),
        "all_hidden_axes_unique": len(set(hidden_canonicals)) == len(evaluations),
    }
    summary = {
        "format": "polaris-blind-composite-axis-multiseed-gate-v1",
        "status": "synthetic_blind_composite_axis_not_model_intelligence",
        "passed": all(bool(item["passed"]) for item in evaluations)
        and coverage["all_hidden_axes_unique"],
        "seed_count": len(SEEDS),
        "seeds_passed": sum(bool(item["passed"]) for item in evaluations),
        "elapsed_seconds": time.perf_counter() - started,
        "process_isolation": {
            "generator_process": True,
            "reframer_process": True,
            "evaluator_process": True,
            "holdout_cli_argument_exposed_to_reframer": False,
        },
        "coverage": coverage,
        "evaluations": evaluations,
        "truth_boundary": {
            "gpu_used": False,
            "model_started": False,
            "training_large_neural_model": False,
            "search_grammar_known_in_advance": True,
            "hidden_expression_known_to_reframer": False,
            "heldout_outcomes_known_to_reframer": False,
        },
    }
    output = root / "blind_composite_gate_receipt.json"
    write_json(output, summary)
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "seeds": f"{summary['seeds_passed']}/{summary['seed_count']}",
                "elapsed_seconds": round(float(summary["elapsed_seconds"]), 4),
                "receipt": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
