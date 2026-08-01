"""纯CPU合成自检：覆盖训练、block自由滚动、拒绝回放、成本与UTF-8。"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from acceptance_simulator import simulate
from cost_model import apply_stop_gate, architecture_cost
from draft_core import (
    CascadedBlockHead,
    DraftDataset,
    ShortlistConfig,
    build_shortlist,
    load_contract,
    sha256_file,
    token_keys,
    write_jsonl,
)
from train_future_head import train


HERE = Path(__file__).resolve().parent


def synthetic_fixture() -> tuple[DraftDataset, dict, CascadedBlockHead, dict]:
    contract = copy.deepcopy(load_contract())
    contract["hidden_size"] = 32
    contract["vocab_size"] = 64
    contract["future_head"]["rank"] = 8
    contract["shortlist"] = {
        "native_top_k": 4,
        "recent_tokens": 4,
        "train_frequent": 4,
        "candidate_limit": 8,
        "priority": contract["shortlist"]["priority"],
    }
    rng = np.random.default_rng(4701)
    count = 144
    hidden = rng.normal(size=(count, 32)).astype(np.float32)
    native_ids = np.tile(np.array([0, 1, 2, 3], dtype=np.int32), (count, 1))
    native_logits = np.tile(np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32), (count, 1))
    teacher = CascadedBlockHead.initialize(32, 8, 3, 991)
    shortlist_config = ShortlistConfig.from_contract(contract)
    candidates = build_shortlist(native_ids[0], [4, 5, 6, 7], [], shortlist_config)
    keys = token_keys(candidates, 8)
    states = teacher.states(hidden)
    rows = []
    for record in range(count):
        if record < 64:
            split = "train"
        elif record < 104:
            split = "validation"
        else:
            split = "test"
        future = [int(candidates[int(np.argmax(keys @ states[record, position]))]) for position in range(3)]
        validator = [0] + future
        rows.append(
            {
                "record": record,
                "anchor_id": f"synthetic-{record:04d}",
                "trajectory_id": f"trajectory-{record:04d}",
                "group_id": f"{split}-group-{record:04d}",
                "template_cluster_id": f"{split}-template-{record:04d}",
                "split": split,
                "context_bucket": ["lt_2k", "2k_8k", "gt_8k"][record % 3],
                "recent_token_ids": [4, 5, 6, 7],
                "validator_token_ids": validator,
                "validator_terminated": False,
                "validator_source": "v38-free-greedy-from-anchor",
                "oracle_token_ids": validator,
                "oracle_terminated": False,
            }
        )
    manifest = {
        "format": "colorlm-v47-parallel-draft-dataset-manifest-v1",
        "base_model": "ColorLM-v38-Qwen36-Shared-Sequence-Policy",
        "rows": {"count": count},
        "capture": {
            "mode": "one-anchor-free-greedy-rollout",
            "temperature": 0,
            "validator": "v38-native",
            "first_token_native_logits": True,
        },
        "synthetic_fixture": True,
    }
    dataset = DraftDataset(manifest, rows, hidden, native_ids, native_logits, Path("synthetic"))
    dataset.validate()
    metadata = {
        "format": "colorlm-v47-cascaded-block-head-v1",
        "first_token": "v38-native-logits-untrained",
        "future_positions": [2, 3, 4],
        "rank": 8,
        "frequent_train_token_ids": [1, 2, 3, 4],
    }
    return dataset, contract, teacher, metadata


def check_utf8_no_bom() -> list[str]:
    errors = []
    for path in HERE.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".json"}:
            continue
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            errors.append(f"BOM: {path.name}")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"非UTF-8: {path.name}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "selftest_report.json")
    args = parser.parse_args()
    dataset, contract, teacher, metadata = synthetic_fixture()
    tests = 0

    assert dataset.native_top_ids.shape == (144, 4)
    tests += 1
    assert all(row["validator_token_ids"][0] == 0 for row in dataset.rows)
    tests += 1
    candidates = build_shortlist(
        dataset.native_top_ids[0], dataset.rows[0]["recent_token_ids"], [], ShortlistConfig.from_contract(contract)
    )
    assert candidates == [0, 1, 2, 3, 7, 6, 5, 4]
    tests += 1
    keys = token_keys(candidates, 8)
    assert keys.shape == (8, 8) and np.allclose(np.linalg.norm(keys, axis=1), 1.0)
    tests += 1

    trained, fit = train(dataset, contract, epochs=60, learning_rate=0.03, seed=12)
    assert fit["first_token_trainable"] is False and fit["trained_positions"] == [2, 3, 4]
    tests += 1
    assert fit["final_train"]["cross_entropy"] < fit["initial_train"]["cross_entropy"]
    tests += 1
    assert trained.input_weight.shape == (32, 8) and trained.bias.shape == (3, 8)
    tests += 1

    perfect_report, perfect_replay = simulate(dataset, teacher, metadata, contract)
    assert perfect_report["coverage"]["gate_passed"] is True
    tests += 1
    assert perfect_report["simulation_executed"] is True
    tests += 1
    assert perfect_report["free_roll"]["evaluation"]["mean_accepted_draft_tokens"] == 4.0
    tests += 1
    assert not perfect_replay
    tests += 1

    zero = CascadedBlockHead(
        np.zeros_like(teacher.input_weight),
        np.zeros_like(teacher.cascade_weight),
        np.zeros_like(teacher.bias),
        teacher.rank,
    )
    rejected_report, rejected_replay = simulate(dataset, zero, metadata, contract)
    assert rejected_report["simulation_executed"] and rejected_replay
    tests += 1
    assert all(row["error_type"] in {"ranking", "shortlist_coverage"} for row in rejected_replay)
    tests += 1

    cost = architecture_cost(contract)
    assert cost["first_token"]["extra_vocabulary_projection_flops"] == 0
    tests += 1
    assert cost["serial_future_head"]["proposal_dependent_scoring_stages"] == 3
    tests += 1
    assert cost["cascaded_block_head"]["proposal_dependent_scoring_stages"] == 1
    tests += 1
    stop_gate = apply_stop_gate(cost, perfect_report, contract)
    assert stop_gate["passed"] is True and stop_gate["analytical_speedup_lower_bound"] >= 1.08
    tests += 1
    missing_gate = apply_stop_gate(cost, None, contract)
    assert missing_gate["passed"] is False
    tests += 1

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        model_path = temporary / "head.npz"
        teacher.save(model_path, metadata)
        loaded, loaded_metadata = CascadedBlockHead.load(model_path)
        assert loaded.rank == teacher.rank and loaded_metadata["first_token"].endswith("untrained")
        rows_path = temporary / "rows.jsonl"
        arrays_path = temporary / "arrays.npz"
        manifest_path = temporary / "manifest.json"
        write_jsonl(rows_path, dataset.rows)
        np.savez(
            arrays_path,
            hidden=dataset.hidden,
            native_top_ids=dataset.native_top_ids,
            native_top_logits=dataset.native_top_logits,
        )
        disk_manifest = copy.deepcopy(dataset.manifest)
        disk_manifest["rows"] = {"path": rows_path.name, "sha256": sha256_file(rows_path), "count": len(dataset.rows)}
        disk_manifest["arrays"] = {"path": arrays_path.name, "sha256": sha256_file(arrays_path)}
        manifest_path.write_text(json.dumps(disk_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reloaded = DraftDataset.load(manifest_path)
        assert reloaded.hidden.shape == dataset.hidden.shape and len(reloaded.rows) == len(dataset.rows)
        tests += 1
    tests += 1
    utf8_errors = check_utf8_no_bom()
    assert not utf8_errors, utf8_errors
    tests += 1

    report = {
        "format": "colorlm-v47-parallel-draft-selftest-v1",
        "passed": True,
        "tests": tests,
        "cpu_only": True,
        "gpu_used": False,
        "synthetic_only": True,
        "synthetic_anchors": len(dataset.rows),
        "fit_loss_before": fit["initial_train"]["cross_entropy"],
        "fit_loss_after": fit["final_train"]["cross_entropy"],
        "perfect_mean_acceptance": perfect_report["free_roll"]["evaluation"]["mean_accepted_draft_tokens"],
        "claim_limit": "合成自检只证明代码路径和停止门可运行，不是ColorLM真实接受率。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
