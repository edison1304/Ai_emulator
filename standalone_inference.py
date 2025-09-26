#!/usr/bin/env python3
"""
Standalone inference script that mimics the inference generation from training loop.
This uses the same data loading and inference logic as the training process.
"""

import argparse
import contextlib
import logging
import os
from pathlib import Path

import torch
import dacite

import fme
import fme.core.logging_utils as logging_utils
from fme.core.cli import prepare_config, prepare_directory
from fme.core.generics.inference import run_inference
from fme.core.generics.trainer import inference_one_epoch
from fme.core.timing import GlobalTimer
from fme.ace.stepper import load_stepper, load_stepper_config
from fme.ace.data_loading.inference import InferenceDataLoaderConfig
from fme.ace.aggregator.inference import InferenceAggregatorConfig


def main():
    parser = argparse.ArgumentParser(description="Standalone inference using training-style data loading")
    parser.add_argument("config", help="Path to training config YAML file")
    parser.add_argument("--checkpoint", help="Path to checkpoint file (default: best checkpoint from config)")
    parser.add_argument("--output-dir", help="Output directory for inference results")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch number for logging")
    args = parser.parse_args()

    # Load config
    config_data = prepare_config(args.config)
    
    # Setup logging
    output_dir = args.output_dir or "inference_output"
    prepare_directory(output_dir, config_data)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "inference.log")),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting standalone inference")
    logging.info(f"Using config: {args.config}")
    logging.info(f"Output directory: {output_dir}")
    
    # Load stepper and config
    checkpoint_path = args.checkpoint or os.path.join(config_data["experiment_dir"], "training_checkpoints", "best_ckpt.tar")
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    
    stepper = load_stepper(checkpoint_path)
    stepper_config = load_stepper_config(checkpoint_path)
    
    # Setup inference data loader (same as training)
    from fme.ace.data_loading.inference import InferenceInitialConditionIndices
    
    # Convert dict to proper config objects
    start_indices = dacite.from_dict(
        data_class=InferenceInitialConditionIndices,
        data=config_data["inference"]["loader"]["start_indices"],
        config=dacite.Config(strict=True)
    )
    
    # Use the same approach as training - let the config system handle the dataset conversion
    inference_config = InferenceDataLoaderConfig(
        dataset=config_data["train_loader"]["dataset"],  # Pass dict directly
        start_indices=start_indices
    )
    
    # Create inference data
    logging.info("Loading inference data")
    inference_data = inference_config.build()
    
    # Setup aggregator
    aggregator_config = InferenceAggregatorConfig()
    aggregator = aggregator_config.build(
        dataset_info=inference_data.dataset_info,
        n_timesteps=config_data["inference"]["n_forward_steps"],
        output_dir=output_dir
    )
    
    # Validation context (same as training)
    def validation_context():
        return contextlib.nullcontext()
    
    # Run inference (same as training loop)
    logging.info("Running inference")
    with GlobalTimer():
        logs = inference_one_epoch(
            stepper=stepper,
            validation_context=validation_context,
            dataset=inference_data,
            aggregator=aggregator,
            label="inference",
            epoch=args.epoch
        )
    
    logging.info("Inference completed")
    logging.info(f"Results: {logs}")
    
    # Flush diagnostics
    aggregator.flush_diagnostics(subdir=f"epoch_{args.epoch:04d}")
    
    logging.info(f"Inference results saved to: {output_dir}")


if __name__ == "__main__":
    main()