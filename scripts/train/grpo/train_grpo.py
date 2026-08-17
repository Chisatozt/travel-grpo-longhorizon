#!/usr/bin/env python3
"""Compatibility CLI for the project GRPO launcher."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.training.grpo.launcher import (  # noqa: E402
    hydra_overrides,
    load_profile,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
