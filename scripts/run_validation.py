#!/usr/bin/env python3
"""Run validation only against an existing checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Sequence

import dacite

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fme
from fme.ace.train.train import build_trainer
from fme.ace.train.train_config import TrainBuilders, TrainConfig
from fme.core.cli import prepare_config, prepare_directory
import fme.core.logging_utils as logging_utils
from fme.core.dicts import to_flat_dict


def _build_config(yaml_config: str, override: Sequence[str] | None) -> tuple[TrainConfig, dict]:
    """Load the YAML config and return both the dataclass and raw dict."""

    config_dict = prepare_config(yaml_config, override)
    config = dacite.from_dict(
        data_class=TrainConfig,
        data=config_dict,
        config=dacite.Config(strict=True),
    )
    return config, config_dict


def run_validation(yaml_config: str, override: Sequence[str] | None = None) -> None:
    config, raw_config = _build_config(yaml_config, override)
    prepare_directory(config.experiment_dir, raw_config)

    # Set up logging similar to the training entrypoint.
    config.logging.configure_logging(config.experiment_dir, log_filename="out.log")
    env_vars = logging_utils.retrieve_env_vars()
    logging_utils.log_versions()

    if fme.using_gpu():
        import torch

        logging.info(
            "Using CUDA device %s",
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )

    config.logging.configure_wandb(
        config=to_flat_dict(dataclasses.asdict(config)),
        env_vars=env_vars,
        notes=logging_utils.log_beaker_url(),
    )

    builders = TrainBuilders(config)
    trainer = build_trainer(builders, config)

    logging.info("Running validation using checkpoint at %s", getattr(config.stepper, "checkpoint_path", "<unknown>"))
    logs = trainer.validate_one_epoch()

    # Convert tensors to floats for logging clarity.
    def _to_numeric(value):
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return value.item()
        except Exception:
            pass
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        return value

    logs = {k: _to_numeric(v) for k, v in logs.items()}
    logging.info("Validation metrics: %s", logs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a checkpoint without training")
    parser.add_argument("yaml_config", type=str, help="Path to the validation config file")
    parser.add_argument(
        "--override",
        nargs="*",
        help="Optional dotlist overrides (e.g. stepper.checkpoint_path=/path/to/ckpt.tar)",
    )
    args = parser.parse_args()
    run_validation(args.yaml_config, args.override)


if __name__ == "__main__":
    main()
