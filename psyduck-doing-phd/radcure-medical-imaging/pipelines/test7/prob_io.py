"""Load / save Test7 probability dumps (raw nnUNet .npz and slim crop)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SLIM_SUFFIX = ".slim.npz"
SLIM_FORMAT_VERSION = 1


@dataclass
class SlimProbabilities:
    """
    Cropped soft predictions around the tumor ROI.

    Arrays are in **crop** coordinates. ``bbox`` is half-open
    ``(x0, x1, y0, y1, z0, z1)`` into the full volume of ``full_shape``.
    """

    p_gtvp: np.ndarray  # (Xc, Yc, Zc) float16/float32
    class_indices: np.ndarray  # (K,) int — includes gtvp_index
    p_classes: np.ndarray  # (K, Xc, Yc, Zc) aligned with class_indices
    bbox: Tuple[int, int, int, int, int, int]
    full_shape: Tuple[int, int, int]
    gtvp_index: int
    case_id: str = ""

    @property
    def crop_shape(self) -> Tuple[int, int, int]:
        return tuple(int(x) for x in self.p_gtvp.shape)  # type: ignore[return-value]

    def index_of(self, class_idx: int) -> Optional[int]:
        hits = np.where(self.class_indices == int(class_idx))[0]
        return int(hits[0]) if len(hits) else None

    def p_class_crop(self, class_idx: int) -> Optional[np.ndarray]:
        """Probability map for ``class_idx`` in crop coords, or None if not stored."""
        if int(class_idx) == int(self.gtvp_index):
            return np.asarray(self.p_gtvp, dtype=np.float32)
        j = self.index_of(class_idx)
        if j is None:
            return None
        return np.asarray(self.p_classes[j], dtype=np.float32)

    def channel_stack_crop(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ``(probs_k, class_indices)`` with shape ``(K, Xc, Yc, Zc)``.

        Prefer ``p_classes`` if it already contains GTVp; else prepend ``p_gtvp``.
        """
        idxs = np.asarray(self.class_indices, dtype=np.int32)
        probs = np.asarray(self.p_classes, dtype=np.float32)
        if self.index_of(self.gtvp_index) is None:
            probs = np.concatenate(
                [np.asarray(self.p_gtvp, dtype=np.float32)[None, ...], probs],
                axis=0,
            )
            idxs = np.concatenate(
                [np.array([self.gtvp_index], dtype=np.int32), idxs]
            )
        return probs, idxs

    def crop_gt(self, gt_full: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1, z0, z1 = self.bbox
        return gt_full[x0:x1, y0:y1, z0:z1]

    def crop_volume(self, vol: np.ndarray) -> np.ndarray:
        """Crop a full-volume array (X,Y,Z) or (C,X,Y,Z) with spatial dims last-3."""
        x0, x1, y0, y1, z0, z1 = self.bbox
        if vol.ndim == 3:
            return vol[x0:x1, y0:y1, z0:z1]
        if vol.ndim == 4:
            return vol[:, x0:x1, y0:y1, z0:z1]
        raise ValueError(f"Expected 3D/4D volume, got {vol.shape}")


def load_probability_npz(path: Path) -> np.ndarray:
    """
    Load softmax probabilities from nnUNetv2 ``--save_probabilities`` output.

    Expected shape: ``(C, X, Y, Z)`` float32 in [0, 1].

    Note: spatial axis order may still be nnUNet's internal order (often a
    transpose of the on-disk NIfTI). Use ``align_probs_to_reference`` before
    comparing with GT / CT loaded via nibabel.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Probability file not found: {path}")

    with np.load(path) as data:
        keys = list(data.keys())
        if "probabilities" in data:
            probs = np.asarray(data["probabilities"])
        elif "softmax" in data:
            probs = np.asarray(data["softmax"])
        else:
            probs = None
            for k in keys:
                arr = np.asarray(data[k])
                if arr.ndim == 4:
                    probs = arr
                    break
            if probs is None:
                raise KeyError(f"No probability array in {path}; keys={keys}")

    if probs.ndim != 4:
        raise ValueError(
            f"Expected probabilities shape (C,X,Y,Z), got {probs.shape} in {path}"
        )
    return probs.astype(np.float32, copy=False)


def align_probs_to_reference(
    probs: np.ndarray,
    reference_shape: Sequence[int],
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    """
    Transpose spatial axes of ``probs`` ``(C, …)`` so they match ``reference_shape``.

    nnUNet ``.npz`` dumps often keep the network/export axis order (e.g.
    ``(Z, X, Y)``) while nibabel GT/CT are ``(X, Y, Z)``.

    Returns
    -------
    probs_aligned : np.ndarray
        Shape ``(C, *reference_shape)``.
    spatial_transpose : tuple[int,int,int] | None
        Permutation applied to the three spatial axes, or None if already matched.
    """
    if probs.ndim != 4:
        raise ValueError(f"Expected probs (C,X,Y,Z), got {probs.shape}")
    ref = tuple(int(x) for x in reference_shape)
    if len(ref) != 3:
        raise ValueError(f"reference_shape must be length 3, got {ref}")

    spatial = tuple(int(x) for x in probs.shape[1:])
    if spatial == ref:
        return probs, None

    from itertools import permutations

    for perm in permutations((0, 1, 2)):
        trial = (spatial[perm[0]], spatial[perm[1]], spatial[perm[2]])
        if trial == ref:
            # probs axes: 0=C, 1..3=spatial → new order (0, 1+perm0, 1+perm1, 1+perm2)
            axes = (0, 1 + perm[0], 1 + perm[1], 1 + perm[2])
            return np.transpose(probs, axes), perm

    raise ValueError(
        f"Cannot align probs spatial {spatial} to reference {ref} "
        "(not a permutation of the same sizes)"
    )


def slim_path_for_case(prob_dir: Path, case_id: str) -> Path:
    return Path(prob_dir) / f"{case_id}{SLIM_SUFFIX}"


def find_slim_file(prob_dir: Path, case_id: str) -> Optional[Path]:
    p = slim_path_for_case(prob_dir, case_id)
    return p if p.is_file() else None


def find_raw_npz(prob_dir: Path, case_id: str) -> Optional[Path]:
    candidates = [
        Path(prob_dir) / f"{case_id}.npz",
        Path(prob_dir) / f"{case_id}.npz.npz",
    ]
    for c in candidates:
        if c.is_file():
            return c
    matches = list(Path(prob_dir).glob(f"**/{case_id}.npz"))
    return matches[0] if matches else None


def find_probability_file(prob_dir: Path, case_id: str) -> Optional[Path]:
    """Prefer slim crop; fall back to raw nnUNet ``.npz``."""
    slim = find_slim_file(prob_dir, case_id)
    if slim is not None:
        return slim
    return find_raw_npz(prob_dir, case_id)


def list_cases_with_probabilities(prob_dir: Path) -> list[str]:
    """Case ids that have slim and/or raw probability dumps (prefer unique ids)."""
    prob_dir = Path(prob_dir)
    cases: set[str] = set()
    for p in prob_dir.glob(f"*{SLIM_SUFFIX}"):
        # case_0405.slim.npz → stem case_0405.slim → need case_0405
        name = p.name
        if name.endswith(SLIM_SUFFIX):
            cases.add(name[: -len(SLIM_SUFFIX)])
    for p in sorted(prob_dir.glob("*.npz")):
        name = p.name
        if name.endswith(SLIM_SUFFIX):
            continue
        if name.endswith(".npz.npz"):
            cases.add(name[: -len(".npz.npz")])
        else:
            cases.add(p.stem)
    return sorted(cases)


def save_slim_npz(path: Path, slim: SlimProbabilities, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": np.int16(SLIM_FORMAT_VERSION),
        "p_gtvp": np.asarray(slim.p_gtvp, dtype=np.float16),
        "p_classes": np.asarray(slim.p_classes, dtype=np.float16),
        "class_indices": np.asarray(slim.class_indices, dtype=np.int16),
        "bbox": np.asarray(slim.bbox, dtype=np.int32),
        "full_shape": np.asarray(slim.full_shape, dtype=np.int32),
        "gtvp_index": np.int16(slim.gtvp_index),
    }
    for k, v in meta.items():
        if isinstance(v, (str, bytes)):
            payload[k] = np.asarray(v)
        else:
            payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)


def load_slim_npz(path: Path) -> SlimProbabilities:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        bbox = tuple(int(x) for x in np.asarray(data["bbox"]).tolist())
        full_shape = tuple(int(x) for x in np.asarray(data["full_shape"]).tolist())
        if len(bbox) != 6 or len(full_shape) != 3:
            raise ValueError(f"Bad slim geometry in {path}")
        return SlimProbabilities(
            p_gtvp=np.asarray(data["p_gtvp"]),
            class_indices=np.asarray(data["class_indices"]),
            p_classes=np.asarray(data["p_classes"]),
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]),
            full_shape=(full_shape[0], full_shape[1], full_shape[2]),
            gtvp_index=int(np.asarray(data["gtvp_index"]).item()),
            case_id=path.name[: -len(SLIM_SUFFIX)]
            if path.name.endswith(SLIM_SUFFIX)
            else path.stem,
        )


def load_case_probabilities(
    prob_dir: Path, case_id: str
) -> Tuple[str, object]:
    """
    Load preferred dump for a case.

    Returns ``(\"slim\", SlimProbabilities)`` or ``(\"raw\", ndarray (C,X,Y,Z))``.
    """
    slim_p = find_slim_file(prob_dir, case_id)
    if slim_p is not None:
        return "slim", load_slim_npz(slim_p)
    raw_p = find_raw_npz(prob_dir, case_id)
    if raw_p is not None:
        return "raw", load_probability_npz(raw_p)
    raise FileNotFoundError(
        f"No slim or raw probabilities for {case_id} in {prob_dir}"
    )


def dilate_binary(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1, 1:-1]
            | padded[:-2, 1:-1, 1:-1]
            | padded[2:, 1:-1, 1:-1]
            | padded[1:-1, :-2, 1:-1]
            | padded[1:-1, 2:, 1:-1]
            | padded[1:-1, 1:-1, :-2]
            | padded[1:-1, 1:-1, 2:]
        )
    return out


def bbox_from_mask(
    mask: np.ndarray,
    margin: int = 8,
    full_shape: Optional[Sequence[int]] = None,
) -> Tuple[int, int, int, int, int, int]:
    """Half-open bbox with margin, clipped to volume."""
    coords = np.where(mask)
    shape = tuple(full_shape) if full_shape is not None else mask.shape
    if coords[0].size == 0:
        # Degenerate: tiny center crop
        cx, cy, cz = [s // 2 for s in shape]
        m = max(margin, 4)
        return (
            max(0, cx - m),
            min(shape[0], cx + m),
            max(0, cy - m),
            min(shape[1], cy + m),
            max(0, cz - m),
            min(shape[2], cz + m),
        )
    x0 = max(0, int(coords[0].min()) - margin)
    x1 = min(shape[0], int(coords[0].max()) + 1 + margin)
    y0 = max(0, int(coords[1].min()) - margin)
    y1 = min(shape[1], int(coords[1].max()) + 1 + margin)
    z0 = max(0, int(coords[2].min()) - margin)
    z1 = min(shape[2], int(coords[2].max()) + 1 + margin)
    return x0, x1, y0, y1, z0, z1


def tumor_roi_mask(
    gt: Optional[np.ndarray],
    probs: np.ndarray,
    gtvp_index: int,
    dilate_iter: int = 2,
    pred_threshold: float = 0.3,
) -> np.ndarray:
    """
    Binary ROI used to define the crop.

    Prefer dilated GT GTVp; if absent, use ``P(GTVp) >= pred_threshold``,
    then argmax==GTVp.
    """
    spatial = probs.shape[1:]
    if gt is not None and gt.shape == spatial and np.any(gt == gtvp_index):
        return dilate_binary(gt == gtvp_index, iterations=dilate_iter)

    p = probs[gtvp_index]
    soft = p >= pred_threshold
    if np.any(soft):
        return dilate_binary(soft, iterations=dilate_iter)

    hard = np.argmax(probs, axis=0) == gtvp_index
    return dilate_binary(hard, iterations=max(dilate_iter, 1))


def select_class_indices(
    probs_crop: np.ndarray,
    gt_crop: Optional[np.ndarray],
    gtvp_index: int,
    organ_dict: Dict[str, int],
    always_include: Iterable[str] = ("GTVn",),
    top_k: int = 12,
    exclude_names: Optional[Iterable[str]] = None,
) -> List[int]:
    """
    Classes to keep in the slim dump (always includes GTVp).

    Priority:
    1. GTVp (+ always_include, e.g. GTVn)
    2. Labels present in the GT crop (competing organs)
    3. top_k by mean probability **inside the GTVp-support ROI**
       (not the whole crop — whole-crop mean prefers head/skull/mandible)
    """
    from pipelines.test7.paths import competing_region_names

    exclude = set(
        exclude_names
        or (
            "background",
            "anatomical_region",
            "other-tissue",
            # Large FOV fillers — still available if present in GT crop
            "head",
            "skull",
            "mandible",
            "teeth_upper",
            "teeth_lower",
        )
    )
    selected: List[int] = [int(gtvp_index)]

    def _add(idx: int) -> None:
        idx = int(idx)
        if idx < 0 or idx >= probs_crop.shape[0]:
            return
        if idx not in selected:
            selected.append(idx)

    for name in always_include:
        if name in organ_dict:
            _add(int(organ_dict[name]))

    allowed = set(competing_region_names(organ_dict))
    name_by_idx = {v: k for k, v in organ_dict.items() if isinstance(v, int)}

    if gt_crop is not None:
        for idx in np.unique(gt_crop):
            idx = int(idx)
            name = name_by_idx.get(idx)
            if name is None or name in ("background", "anatomical_region", "other-tissue"):
                continue
            if name == "GTVp" or name in allowed or name in always_include:
                _add(idx)

    # Score top-k inside GTVp support (GT or soft), not whole crop
    p_gtvp = probs_crop[int(gtvp_index)]
    support = p_gtvp >= 0.05
    if gt_crop is not None:
        support = support | (gt_crop == int(gtvp_index))
    if not np.any(support):
        support = np.ones(p_gtvp.shape, dtype=bool)

    # Mean P within support; ignore excluded bulk fillers for top-k
    scores = np.zeros(probs_crop.shape[0], dtype=np.float64)
    for idx in range(probs_crop.shape[0]):
        name = name_by_idx.get(idx, "")
        if name in exclude:
            continue
        scores[idx] = float(probs_crop[idx][support].mean())

    order = np.argsort(-scores)
    added = 0
    for idx in order:
        idx = int(idx)
        if idx in selected:
            continue
        if scores[idx] <= 0:
            continue
        name = name_by_idx.get(idx, "")
        if name in exclude:
            continue
        _add(idx)
        added += 1
        if added >= top_k:
            break

    return selected


def build_slim_from_full(
    probs: np.ndarray,
    gtvp_index: int,
    class_indices: Sequence[int],
    bbox: Tuple[int, int, int, int, int, int],
    case_id: str = "",
) -> SlimProbabilities:
    x0, x1, y0, y1, z0, z1 = bbox
    crop = probs[:, x0:x1, y0:y1, z0:z1]
    idxs = [int(i) for i in class_indices]
    if gtvp_index not in idxs:
        idxs = [int(gtvp_index)] + idxs
    p_classes = np.stack([crop[i] for i in idxs], axis=0)
    return SlimProbabilities(
        p_gtvp=np.asarray(crop[gtvp_index], dtype=np.float16),
        class_indices=np.asarray(idxs, dtype=np.int16),
        p_classes=np.asarray(p_classes, dtype=np.float16),
        bbox=bbox,
        full_shape=tuple(int(s) for s in probs.shape[1:]),  # type: ignore[arg-type]
        gtvp_index=int(gtvp_index),
        case_id=case_id,
    )
