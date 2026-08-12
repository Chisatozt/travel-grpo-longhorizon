#!/usr/bin/env python3
"""Apply hash-checked project connections to the pinned veRL 0.8 trainer."""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import find_spec
from importlib.metadata import version
from pathlib import Path

VERL_VERSION = "0.8.0"
SOURCE_SHA256 = "DE58D295CF86656A28196B0718168D4A11666F3E30957B7E166914496C2A6D66"
LEGACY_DYNAMIC_PATCHED_SHA256 = "84C334738B82ABA8B57A2D735DD0C17CC48C6D5852247E12546CBC7987C7DC36"
PATCHED_SHA256 = "51E774CC9E112EEE00EBBEDDAB99FBF9D89C34C900F4473E1254E9CA8637CF64"
PATCH_PAYLOAD_SHA256 = "0E847FEF13300985C60E34DCF16B2FFF87B7E9CA0DDC8274223E9BFDE854B375"
TURN_PATCH_PAYLOAD_SHA256 = "3321AC2DC15E95FD2AB9E5AE60E72B9C9AF2458BBAD76A9CE6B14EEAC4473FBE"

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
BEFORE_TURN_CREDIT = '''                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
'''
AFTER_TURN_CREDIT = BEFORE_TURN_CREDIT + '''                        travel_turn_credit = self.config.algorithm.get("travel_turn_credit", {})
                        if travel_turn_credit.get("mode", "off") == "train":
                            from travel_grpo.training.grpo.turn_credit import reshape_batch_advantages

                            batch = reshape_batch_advantages(batch, self.config.algorithm)
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
        print(f"veRL project connection patches are present: {path}")
        return 0
    if current not in {SOURCE_SHA256, LEGACY_DYNAMIC_PATCHED_SHA256}:
        raise RuntimeError(f"unknown veRL trainer SHA-256 {current}: {path}")
    payload = (BEFORE_INSTALL + AFTER_INSTALL + BEFORE_SKIP + AFTER_SKIP).encode()
    if digest(payload) != PATCH_PAYLOAD_SHA256:
        raise RuntimeError("internal patch payload SHA-256 mismatch")
    turn_payload = (BEFORE_TURN_CREDIT + AFTER_TURN_CREDIT).encode()
    if digest(turn_payload) != TURN_PATCH_PAYLOAD_SHA256:
        raise RuntimeError("internal turn-credit patch payload SHA-256 mismatch")
    text = raw.decode("utf-8")
    if current == SOURCE_SHA256:
        if text.count(BEFORE_INSTALL) != 1 or text.count(BEFORE_SKIP) != 1:
            raise RuntimeError("veRL dynamic-sampling patch targets are ambiguous or missing")
        text = text.replace(BEFORE_INSTALL, AFTER_INSTALL).replace(
            BEFORE_SKIP, AFTER_SKIP
        )
    if text.count(BEFORE_TURN_CREDIT) != 1:
        raise RuntimeError("veRL turn-credit patch target is ambiguous or missing")
    patched = text.replace(BEFORE_TURN_CREDIT, AFTER_TURN_CREDIT).encode()
    if digest(patched) != PATCHED_SHA256:
        raise RuntimeError("patched veRL trainer SHA-256 does not match the pinned result")
    if args.check:
        raise RuntimeError("veRL project connection patches are not installed")
    path.write_bytes(patched)
    print(f"patched veRL 0.8 trainer: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
