"""
Stable name→RGBA colours for combined masks.

Rule: GTVp = red, GTVn = pink; TotalSegmentator organs never use those hues.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

RGBA = Tuple[float, float, float, float]

# Fixed special labels (alpha for overlays)
COLOR_BACKGROUND: RGBA = (0.0, 0.0, 0.0, 0.0)
COLOR_ANATOMICAL: RGBA = (0.15, 0.75, 0.85, 0.45)  # cyan
COLOR_OTHER_TISSUE: RGBA = (0.55, 0.55, 0.55, 0.50)  # gray
COLOR_GTVP: RGBA = (1.0, 0.0, 0.0, 0.85)  # red
COLOR_GTVN: RGBA = (1.0, 0.41, 0.71, 0.85)  # pink


def _is_red_or_pink(rgb: Sequence[float], *, red_thr: float = 0.65) -> bool:
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    if r < red_thr:
        return False
    # red-ish or pink-ish (high R, low-mid G, low-mid B or high B)
    return (g < 0.45 and b < 0.45) or (g < 0.55 and b > 0.35)


def _organ_palette(n: int) -> List[RGBA]:
    """Deterministic colours for TS organs, excluding red/pink family."""
    # Hand-picked distinct hues (no pure red / hot pink)
    base: List[RGBA] = [
        (0.12, 0.47, 0.71, 0.70),
        (0.20, 0.63, 0.17, 0.70),
        (0.89, 0.47, 0.04, 0.70),
        (0.58, 0.40, 0.74, 0.70),
        (0.55, 0.34, 0.29, 0.70),
        (0.09, 0.75, 0.81, 0.70),
        (0.74, 0.74, 0.13, 0.70),
        (0.35, 0.35, 0.85, 0.70),
        (0.17, 0.63, 0.52, 0.70),
        (0.80, 0.60, 0.15, 0.70),
        (0.40, 0.76, 0.65, 0.70),
        (0.55, 0.63, 0.80, 0.70),
        (0.99, 0.75, 0.44, 0.70),
        (0.55, 0.83, 0.78, 0.70),
        (0.75, 0.72, 0.85, 0.70),
        (0.98, 0.71, 0.85, 0.55),  # light rose — borderline; filtered below
        (0.70, 0.87, 0.54, 0.70),
        (0.98, 0.88, 0.55, 0.70),
        (0.45, 0.62, 0.81, 0.70),
        (0.99, 0.85, 0.65, 0.70),
        (0.28, 0.47, 0.65, 0.70),
        (0.85, 0.65, 0.35, 0.70),
        (0.45, 0.70, 0.30, 0.70),
        (0.65, 0.35, 0.55, 0.70),
        (0.30, 0.55, 0.55, 0.70),
        (0.80, 0.50, 0.25, 0.70),
        (0.50, 0.50, 0.20, 0.70),
        (0.25, 0.35, 0.60, 0.70),
        (0.60, 0.25, 0.40, 0.70),
        (0.15, 0.55, 0.35, 0.70),
    ]
    safe = [c for c in base if not _is_red_or_pink(c)]
    if not safe:
        safe = [(0.2, 0.5, 0.7, 0.7)]
    out: List[RGBA] = []
    for i in range(n):
        out.append(safe[i % len(safe)])
    return out


def rgba_by_name(organ_dict: Mapping[str, int]) -> Dict[str, RGBA]:
    """
    Stable name → RGBA map from a (canonical) organ dictionary.

    GTVp/GTVn always red/pink; TS organs get non-conflicting palette colours
    in ascending index order (deterministic across cases).
    """
    colors: Dict[str, RGBA] = {
        "background": COLOR_BACKGROUND,
        "anatomical_region": COLOR_ANATOMICAL,
        "other-tissue": COLOR_OTHER_TISSUE,
        "GTVp": COLOR_GTVP,
        "GTVn": COLOR_GTVN,
    }

    reserved = {"background", "anatomical_region", "other-tissue", "GTVp", "GTVn"}
    organ_names = sorted(
        (n for n in organ_dict if n not in reserved),
        key=lambda n: organ_dict[n],
    )
    palette = _organ_palette(len(organ_names))
    for name, rgba in zip(organ_names, palette):
        colors[name] = rgba
    return colors


def rgba_by_index(organ_dict: Mapping[str, int]) -> Dict[int, RGBA]:
    """Index → RGBA using ``rgba_by_name``."""
    by_name = rgba_by_name(organ_dict)
    out: Dict[int, RGBA] = {}
    for name, idx in organ_dict.items():
        out[int(idx)] = by_name.get(name, COLOR_OTHER_TISSUE)
    out[0] = COLOR_BACKGROUND
    return out


def paint_label_rgba(
    label_slice: np.ndarray,
    index_to_rgba: Mapping[int, RGBA],
    *,
    include: Optional[Iterable[int]] = None,
    exclude: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """
    Paint a 2D integer label map into an HxWx4 float overlay.

    Parameters
    ----------
    include
        If set, only these label indices are painted.
    exclude
        Label indices to skip (e.g. hide other-tissue).
    """
    h, w = label_slice.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    include_set = None if include is None else set(include)
    exclude_set = set(exclude or ())
    for idx, rgba in index_to_rgba.items():
        if idx == 0 or idx in exclude_set:
            continue
        if include_set is not None and idx not in include_set:
            continue
        out[label_slice == idx] = rgba
    return out


def organ_indices(
    organ_dict: Mapping[str, int],
    *,
    tumors: bool = False,
    specials: bool = False,
) -> List[int]:
    """
    Indices for TotalSegmentator organs only by default.

    tumors=True also includes GTVp/GTVn; specials=True includes anatomical/other-tissue.
    """
    skip = {"background"}
    if not specials:
        skip |= {"anatomical_region", "other-tissue"}
    if not tumors:
        skip |= {"GTVp", "GTVn"}
    return sorted(
        int(i) for n, i in organ_dict.items() if n not in skip
    )


def save_organ_color_palette(
    organ_dict: Mapping[str, int],
    output_png: str,
    *,
    output_json: Optional[str] = None,
    title: str = "Canonical organ colour palette (reference)",
    ncols: int = 3,
    dpi: int = 150,
) -> str:
    """
    Save a reference figure: colour swatch + index + organ name for every label.

    Also writes optional JSON ``{name: {index, rgb, rgba}}`` next to the PNG
    (or at ``output_json``) so the palette can be reused outside the notebook.

    Background (index 0) is shown as a checkerboard / hollow swatch because it
    is fully transparent in overlays.
    """
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = rgba_by_name(organ_dict)
    # Stable order by label index
    items = sorted(organ_dict.items(), key=lambda kv: int(kv[1]))
    n = len(items)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig_w = 4.2 * ncols
    fig_h = max(2.5, 0.38 * nrows + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=8)

    payload = {}
    for i, (name, idx) in enumerate(items):
        row = i // ncols
        col = i % ncols
        rgba = colors.get(name, COLOR_OTHER_TISSUE)
        rgb = tuple(float(x) for x in rgba[:3])
        alpha = float(rgba[3]) if len(rgba) > 3 else 1.0

        x0, y0 = col + 0.05, row + 0.15
        sw = 0.22
        sh = 0.55
        if name == "background" or alpha < 1e-6:
            # hollow swatch for transparent background
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    sw,
                    sh,
                    fill=False,
                    edgecolor="0.4",
                    linewidth=1.2,
                    linestyle="--",
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    sw,
                    sh,
                    facecolor=rgb + (min(1.0, max(alpha, 0.85)),),
                    edgecolor="0.2",
                    linewidth=0.6,
                )
            )
        ax.text(
            x0 + sw + 0.04,
            y0 + sh / 2,
            f"{int(idx):3d}  {name}",
            va="center",
            ha="left",
            fontsize=7.5,
            family="monospace",
        )
        payload[name] = {
            "index": int(idx),
            "rgb": [round(c, 4) for c in rgb],
            "rgba": [round(c, 4) for c in (rgb + (alpha,))],
        }

    out_png = Path(output_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    json_path = Path(output_json) if output_json else out_png.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(out_png)
