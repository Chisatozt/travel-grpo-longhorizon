#!/usr/bin/env python3
"""Apply the hash-checked veRL 0.8 dynamic-sampling connection patch."""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import find_spec
from importlib.metadata import version
from pathlib import Path

VERL_VERSION = "0.8.0"
SOURCE_SHA256 = "DE58D295CF86656A28196B0718168D4A11666F3E30957B7E166914496C2A6D66"
PATCHED_SHA256 = "84C334738B82ABA8B57A2D735DD0C17CC48C6D5852247E12546CBC7987C7DC36"
PATCH_PAYLOAD_SHA256 = "0E847FEF13300985C60E34DCF16B2FFF87B7E9CA0DDC8274223E9BFDE854B375"

BEFORE_INSTALL = '''        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()
'''
AFTER_INSTALL = BEFORE_INSTALL + '''        travel_sampling = self.config.get("travel_dynamic_sampling", {})
        if travel_sampling.get("enable", False):
            from travel_grpo.training.grpo.dynamic_sampling import install_verl_bounded_sampler

            install_verl_bounded_sampler(self.async_rollout_manager, travel_sampling)
'''
BEFORE_SKIP = '''                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)
'''
AFTER_SKIP = BEFORE_SKIP + '''                        travel_sampling_metrics = combined_gen_output.meta_info.get("travel_dynamic_sampling")
                        if travel_sampling_metrics:
                            logger.log(
                                data={f"sampling/{key}": value for key, value in travel_sampling_metrics.items()},
                                step=self.global_steps,
                            )
                        if combined_gen_output.meta_info.pop("travel_skip_update", False):
                            self.checkpoint_manager.update_weights(self.global_steps)
                            continue
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def trainer_path() -> Path:
    spec = find_spec("verl.trainer.ppo.ray_trainer")
    if spec is None or spec.origin is None:
        raise RuntimeError("verl is not installed")
    return Path(spec.origin).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args()
    found_version = version("verl")
    if found_version != VERL_VERSION:
        raise RuntimeError(f"expected verl=={VERL_VERSION}, found {found_version}")
    path = trainer_path()
    raw = path.read_bytes()
    current = digest(raw)
    if current == PATCHED_SHA256:
        print(f"veRL dynamic-sampling connection patch is present: {path}")
        return 0
    if current != SOURCE_SHA256:
        raise RuntimeError(f"unknown veRL trainer SHA-256 {current}: {path}")
    payload = (BEFORE_INSTALL + AFTER_INSTALL + BEFORE_SKIP + AFTER_SKIP).encode()
    if digest(payload) != PATCH_PAYLOAD_SHA256:
        raise RuntimeError("internal patch payload SHA-256 mismatch")
    text = raw.decode("utf-8")
    if text.count(BEFORE_INSTALL) != 1 or text.count(BEFORE_SKIP) != 1:
        raise RuntimeError("veRL patch targets are ambiguous or missing")
    patched = text.replace(BEFORE_INSTALL, AFTER_INSTALL).replace(BEFORE_SKIP, AFTER_SKIP).encode()
    if digest(patched) != PATCHED_SHA256:
        raise RuntimeError("patched veRL trainer SHA-256 does not match the pinned result")
    if args.check:
        raise RuntimeError("veRL dynamic-sampling connection patch is not installed")
    path.write_bytes(patched)
    print(f"patched veRL 0.8 trainer: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
