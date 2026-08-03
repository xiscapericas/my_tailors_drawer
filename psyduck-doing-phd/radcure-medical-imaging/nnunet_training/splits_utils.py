"""
Ensure nnUNet ``splits_final.json`` exists before training.

``nnUNetv2_plan_and_preprocess`` writes ``.b2nd`` files but often does **not**
create ``splits_final.json`` — nnUNet normally generates it on the first
``nnUNetv2_train`` call. Our pipeline checks splits before launching train,
so we create the file here when it is missing.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Set

from pipelines.radheck.nnunet_split_utils import list_stems_in_split


def splits_final_path(preprocessed_root: str, dataset_name: str) -> str:
    return os.path.join(preprocessed_root, dataset_name, "splits_final.json")


def training_case_ids(dataset_folder: str) -> List[str]:
    """Case stems in imagesTr with paired labels (nnUNet training pool)."""
    return sorted(list_stems_in_split(dataset_folder, "Tr"))


def _generate_splits_nnunet(
    case_ids: List[str], seed: int = 12345, n_splits: int = 5
) -> List[dict]:
    from nnunetv2.utilities.crossval_split import generate_crossval_split

    return generate_crossval_split(case_ids, seed=seed, n_splits=n_splits)


def _filter_splits_to_cases(splits: List[dict], valid: Set[str]) -> List[dict]:
    out: List[dict] = []
    for fold in splits:
        train = [c for c in fold.get("train", []) if c in valid]
        val = [c for c in fold.get("val", []) if c in valid]
        out.append({"train": train, "val": val})
    return out


def ensure_splits_final(
    dataset_folder: str,
    preprocessed_root: str,
    dataset_name: str,
    *,
    reference_preprocessed: Optional[str] = None,
    seed: int = 12345,
    n_splits: int = 5,
) -> str:
    """
    Create ``splits_final.json`` when missing or out of sync with ``imagesTr``.

    Order:
      1. Use existing file if its case pool matches current ``imagesTr``.
      2. Copy from ``NNUNET_SPLITS_REFERENCE`` / ``reference_preprocessed`` and
         keep only cases that exist in the current ``imagesTr`` pool.
      3. Generate a new 5-fold split with nnUNet's ``generate_crossval_split``.
    """
    path = splits_final_path(preprocessed_root, dataset_name)
    case_ids = training_case_ids(dataset_folder)
    if not case_ids:
        raise FileNotFoundError(
            f"No training cases found in {dataset_folder}/imagesTr"
        )
    valid = set(case_ids)

    if os.path.isfile(path):
        with open(path) as f:
            existing = json.load(f)
        in_splits: Set[str] = set()
        for fold in existing:
            in_splits.update(fold.get("train", []) or [])
            in_splits.update(fold.get("val", []) or [])
        extra = in_splits - valid
        missing = valid - in_splits
        if not extra and not missing:
            return path
        print(
            "⚠️  splits_final.json out of sync with imagesTr "
            f"(extra={len(extra)}, missing_from_splits={len(missing)}); regenerating"
        )
        os.remove(path)

    ref_root = reference_preprocessed or os.getenv("NNUNET_SPLITS_REFERENCE")
    if ref_root:
        src = splits_final_path(ref_root, dataset_name)
        if os.path.isfile(src):
            with open(src) as f:
                splits = _filter_splits_to_cases(json.load(f), valid)
            # Only keep a copied split if it still covers the full Tr pool
            covered: Set[str] = set()
            for fold in splits:
                covered.update(fold.get("train", []) or [])
                covered.update(fold.get("val", []) or [])
            if covered == valid and splits and any(f["train"] or f["val"] for f in splits):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump(splits, f, indent=2)
                print(
                    f"✓ Copied and filtered splits_final.json from {src}\n"
                    f"  → {path} ({len(splits)} folds, {len(case_ids)} cases in Tr pool)"
                )
                return path
            print(
                "⚠️  NNUNET_SPLITS_REFERENCE does not cover the full imagesTr pool "
                f"({len(covered)}/{len(valid)}); generating a fresh split instead"
            )

    splits = _generate_splits_nnunet(case_ids, seed=seed, n_splits=n_splits)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    print(
        f"✓ Generated splits_final.json ({len(splits)} folds, "
        f"{len(case_ids)} training cases, seed={seed})\n"
        f"  → {path}"
    )
    return path
