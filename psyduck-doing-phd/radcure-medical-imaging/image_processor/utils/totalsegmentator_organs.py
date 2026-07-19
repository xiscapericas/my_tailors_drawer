"""
Canonical TotalSegmentator organ names for the H&N tasks used in this project.

Source: TotalSegmentator ``map_to_binary`` class maps for the six tasks in
``CaseProcessor.tasks_to_run``. Names match on-disk ``{organ}.nii.gz`` basenames.

This list is **case-independent**: seed the organ dictionary once, then reuse
across all RADCURE / HECKTOR cases so label indices never depend on discovery order.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Same order as CaseProcessor default tasks
DEFAULT_HN_TASKS: Tuple[str, ...] = (
    "head_glands_cavities",
    "head_muscles",
    "headneck_bones_vessels",
    "headneck_muscles",
    "oculomotor_muscles",
    "craniofacial_structures",
)

# Per-task organs in TotalSegmentator class-index order (1..N).
# Duplicate names across tasks (e.g. optic_nerve_*, skull) are intentional in TS;
# the canonical dictionary keeps a single index per unique name.
TASK_ORGANS: Dict[str, Tuple[str, ...]] = {
    "head_glands_cavities": (
        "eye_left",
        "eye_right",
        "eye_lens_left",
        "eye_lens_right",
        "optic_nerve_left",
        "optic_nerve_right",
        "parotid_gland_left",
        "parotid_gland_right",
        "submandibular_gland_right",
        "submandibular_gland_left",
        "nasopharynx",
        "oropharynx",
        "hypopharynx",
        "nasal_cavity_right",
        "nasal_cavity_left",
        "auditory_canal_right",
        "auditory_canal_left",
        "soft_palate",
        "hard_palate",
    ),
    "head_muscles": (
        "masseter_right",
        "masseter_left",
        "temporalis_right",
        "temporalis_left",
        "lateral_pterygoid_right",
        "lateral_pterygoid_left",
        "medial_pterygoid_right",
        "medial_pterygoid_left",
        "tongue",
        "digastric_right",
        "digastric_left",
    ),
    "headneck_bones_vessels": (
        "larynx_air",
        "thyroid_cartilage",
        "hyoid",
        "cricoid_cartilage",
        "zygomatic_arch_right",
        "zygomatic_arch_left",
        "styloid_process_right",
        "styloid_process_left",
        "internal_carotid_artery_right",
        "internal_carotid_artery_left",
        "internal_jugular_vein_right",
        "internal_jugular_vein_left",
    ),
    "headneck_muscles": (
        "sternocleidomastoid_right",
        "sternocleidomastoid_left",
        "superior_pharyngeal_constrictor",
        "middle_pharyngeal_constrictor",
        "inferior_pharyngeal_constrictor",
        "trapezius_right",
        "trapezius_left",
        "platysma_right",
        "platysma_left",
        "levator_scapulae_right",
        "levator_scapulae_left",
        "anterior_scalene_right",
        "anterior_scalene_left",
        "middle_scalene_right",
        "middle_scalene_left",
        "posterior_scalene_right",
        "posterior_scalene_left",
        "sterno_thyroid_right",
        "sterno_thyroid_left",
        "thyrohyoid_right",
        "thyrohyoid_left",
        "prevertebral_right",
        "prevertebral_left",
    ),
    "oculomotor_muscles": (
        "skull",
        "eyeball_right",
        "lateral_rectus_muscle_right",
        "superior_oblique_muscle_right",
        "levator_palpebrae_superioris_right",
        "superior_rectus_muscle_right",
        "medial_rectus_muscle_left",
        "inferior_oblique_muscle_right",
        "inferior_rectus_muscle_right",
        "optic_nerve_left",
        "eyeball_left",
        "lateral_rectus_muscle_left",
        "superior_oblique_muscle_left",
        "levator_palpebrae_superioris_left",
        "superior_rectus_muscle_left",
        "medial_rectus_muscle_right",
        "inferior_oblique_muscle_left",
        "inferior_rectus_muscle_left",
        "optic_nerve_right",
    ),
    "craniofacial_structures": (
        "mandible",
        "teeth_lower",
        "skull",
        "head",
        "sinus_maxillary",
        "sinus_frontal",
        "teeth_upper",
    ),
}


def unique_hn_organ_names(tasks: Tuple[str, ...] = DEFAULT_HN_TASKS) -> List[str]:
    """Ordered unique organ basenames across the given TS tasks (first wins)."""
    seen = set()
    out: List[str] = []
    for task in tasks:
        if task not in TASK_ORGANS:
            raise KeyError(f"Unknown H&N TotalSegmentator task: {task}")
        for name in TASK_ORGANS[task]:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def build_canonical_hn_dictionary(
    *,
    separate_gtvp_gtvn: bool = True,
    tasks: Tuple[str, ...] = DEFAULT_HN_TASKS,
) -> Dict[str, int]:
    """
    Fixed label map: background / anatomical_region / other-tissue → all TS
    organs for ``tasks`` → GTVp → (optional) GTVn.

    Indices do not depend on which organs appear in a given case crop.
    """
    d: Dict[str, int] = {
        "background": 0,
        "anatomical_region": 1,
        "other-tissue": 2,
    }
    idx = 3
    for name in unique_hn_organ_names(tasks):
        d[name] = idx
        idx += 1
    d["GTVp"] = idx
    idx += 1
    if separate_gtvp_gtvn:
        d["GTVn"] = idx
    return d
