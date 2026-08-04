#!/usr/bin/env python3
"""生成发现集，并在 Reframer 运行前封存独立盲测与干预结果。"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from axis_core import (
    CompositeAxis,
    composite_candidates,
    feature_schema,
    schema_snapshot,
    sha256_file,
    write_json,
)


def _features(rng: random.Random, schema: dict[str, tuple[str, str]]) -> dict[str, float]:
    return {
        feature: round(rng.uniform(0.05, 0.95), 8)
        for features in schema.values()
        for feature in features
    }


def _episode(
    rng: random.Random,
    schema: dict[str, tuple[str, str]],
    hidden_axis: CompositeAxis,
    episode_id: str,
) -> dict[str, object]:
    features = _features(rng, schema)
    return {
        "episode_id": episode_id,
        "features": features,
        "outcome": hidden_axis.evaluate(features),
    }


def _balanced_rows(
    *,
    rng: random.Random,
    schema: dict[str, tuple[str, str]],
    hidden_axis: CompositeAxis,
    count: int,
    prefix: str,
) -> list[dict[str, object]]:
    for attempt in range(50):
        rows = [
            _episode(rng, schema, hidden_axis, f"{prefix}-{attempt:02d}-{index:04d}")
            for index in range(count)
        ]
        positive_rate = sum(bool(row["outcome"]) for row in rows) / count
        if 0.43 <= positive_rate <= 0.57:
            return rows
    raise RuntimeError("无法生成近似平衡的数据")


def _interventions(
    *,
    rng: random.Random,
    schema: dict[str, tuple[str, str]],
    hidden_axis: CompositeAxis,
    count: int,
) -> list[dict[str, object]]:
    pairs = []
    for index in range(count):
        before_features = _features(rng, schema)
        selected = hidden_axis.terms[index % len(hidden_axis.terms)]
        after_features = dict(before_features)
        after_features[selected.left], after_features[selected.right] = (
            after_features[selected.right],
            after_features[selected.left],
        )
        before_outcome = hidden_axis.evaluate(before_features)
        after_outcome = hidden_axis.evaluate(after_features)
        if before_outcome == after_outcome:
            raise RuntimeError("干预没有改变隐藏坐标")
        pairs.append(
            {
                "intervention_id": f"iv-{index:04d}",
                "changed_organ": selected.organ_id,
                "before": {"features": before_features, "outcome": before_outcome},
                "after": {"features": after_features, "outcome": after_outcome},
            }
        )
    return pairs


def generate(
    output_dir: Path,
    *,
    seed: int,
    discovery_count: int = 256,
    holdout_count: int = 512,
    intervention_count: int = 64,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = feature_schema()
    rng = random.Random(seed)
    hidden_pool = [
        axis
        for axis in composite_candidates(schema)
        if len(axis.terms) == (2 if seed % 2 else 3)
    ]
    hidden_axis = hidden_pool[rng.randrange(len(hidden_pool))]

    discovery_rows = _balanced_rows(
        rng=rng,
        schema=schema,
        hidden_axis=hidden_axis,
        count=discovery_count,
        prefix="d",
    )
    holdout_rows = _balanced_rows(
        rng=rng,
        schema=schema,
        hidden_axis=hidden_axis,
        count=holdout_count,
        prefix="h",
    )
    interventions = _interventions(
        rng=rng,
        schema=schema,
        hidden_axis=hidden_axis,
        count=intervention_count,
    )

    discovery_path = output_dir / "discovery.json"
    holdout_path = output_dir / "sealed_holdout.json"
    manifest_path = output_dir / "manifest.json"
    write_json(
        discovery_path,
        {
            "format": "polaris-blind-composite-discovery-v1",
            "schema": schema_snapshot(schema),
            "episodes": discovery_rows,
            "contract": {
                "contains_hidden_expression": False,
                "contains_predefined_plane": False,
                "contains_only_raw_organ_measurements_and_outcome": True,
            },
        },
    )
    write_json(
        holdout_path,
        {
            "format": "polaris-blind-composite-sealed-holdout-v1",
            "episodes": holdout_rows,
            "interventions": interventions,
            "private_audit": {
                "seed": seed,
                "hidden_axis": hidden_axis.snapshot(),
            },
        },
    )
    write_json(
        manifest_path,
        {
            "format": "polaris-blind-composite-manifest-v1",
            "seed_commitment": hashlib.sha256(f"polaris:{seed}".encode("utf-8")).hexdigest(),
            "discovery_sha256": sha256_file(discovery_path),
            "sealed_holdout_sha256": sha256_file(holdout_path),
            "discovery_count": discovery_count,
            "holdout_count": holdout_count,
            "intervention_count": intervention_count,
            "sealed_before_synthesis": True,
        },
    )
    return {
        "discovery": discovery_path,
        "holdout": holdout_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    paths = generate(args.output_dir, seed=args.seed)
    print(paths["manifest"])


if __name__ == "__main__":
    main()
