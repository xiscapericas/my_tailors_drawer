#!/usr/bin/env python3
"""
Deprecated entry point — use ``python -m pipelines.test5.transform_cases``.

Keeps old imports working; forwards argv to the unified RADHECK transform
(no anatomy QC, single ``RADHECK_{N}/cases/`` tree).
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "NOTE: pipelines.test5.relabel_tumor_batch is deprecated.\n"
        "      Forwarding to pipelines.test5.transform_cases "
        "(unified RADHECK_{N}/cases, no anatomy QC).\n",
        file=sys.stderr,
    )
    from pipelines.test5.transform_cases import main as transform_main

    transform_main()


if __name__ == "__main__":
    main()
