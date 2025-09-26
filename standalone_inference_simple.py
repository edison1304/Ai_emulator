#!/usr/bin/env python3
"""
Simple standalone inference script that uses the same approach as training loop.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import dacite

import fme
import fme.core.logging_utils as logging_utils
from fme.core.cli import prepare_config, prepare_directory
from fme.core.generics.inference import run_inference, get_record_to_wandb
from fme.core.timing import GlobalTimer
from fme.ace.stepper import load_stepper, load_stepper_config
from fme.ace.data_loading.inference import InferenceDataLoaderConfig
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.evaluator import resolve_variable_metadata
from fme.ace.aggregator.inference import InferenceAggregatorConfig
from fme.ace.data_loading.getters import get_forcing_data


def main():
    parser = argparse.ArgumentParser(description="Simple standalone inference using training-style data loading")
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
    
    # Use the same approach as the training loop - create inference data directly
    logging.info("Loading inference data using training approach")
    
    # Get data requirements
    data_requirements = stepper_config.get_forcing_window_data_requirements(
        n_forward_steps=config_data["inference"]["forward_steps_in_memory"]
    )
    
    # Use the same input variables as training. Prefer values from the config,
    # but fall back to the serialized stepper when the config only contains an
    # inference stub.
    try:
        training_in_names = config_data["stepper"]["step"]["config"]["in_names"]
    except KeyError:
        training_in_names = stepper_config.input_names
    data_requirements.names = training_in_names.copy()
    logging.info(f"[DEBUG] Using training in_names for requirements.names: {data_requirements.names}")
    
    # Create forcing data loader config
    forcing_loader_config = dacite.from_dict(
        data_class=InferenceDataLoaderConfig,
        data=config_data["inference"]["loader"],
        config=dacite.Config(strict=True)
    )
    
    # Create inference dataset directly
    from fme.ace.data_loading.inference import InferenceDataset
    from fme.ace.data_loading.gridded_data import InferenceGriddedData
    from fme.ace.requirements import PrognosticStateDataRequirements
    
    inference_dataset = InferenceDataset(
        config=forcing_loader_config,
        total_forward_steps=config_data["inference"]["n_forward_steps"],
        requirements=data_requirements,
        surface_temperature_name=stepper.surface_temperature_name,
        ocean_fraction_name=stepper.ocean_fraction_name,
    )
    
    # Create data loader
    from torch.utils.data import DataLoader
    
    # InferenceDataset already returns BatchData objects, so no collation needed
    # Set batch_size=None to disable PyTorch's default collation
    loader = DataLoader(inference_dataset, batch_size=None, shuffle=False)
    
    # Create initial condition requirements
    initial_condition_reqs = PrognosticStateDataRequirements(
        names=stepper_config.prognostic_names,
        n_timesteps=1
    )
    
    # Create inference data
    data = InferenceGriddedData(
        loader=loader,
        initial_condition=initial_condition_reqs,
        properties=inference_dataset.properties
    )
    
    # Setup aggregator
    variable_metadata = resolve_variable_metadata(
        dataset_metadata=data.variable_metadata,
        stepper_metadata=stepper.training_variable_metadata,
        stepper_all_names=stepper_config.all_names,
    )
    dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)

    aggregator_config = InferenceAggregatorConfig()
    aggregator = aggregator_config.build(
        dataset_info=dataset_info,
        n_timesteps=config_data["inference"]["n_forward_steps"] + stepper.n_ic_timesteps,
        output_dir=output_dir,
    )

    writer_config = DataWriterConfig(
        save_prediction_files=True,
        save_monthly_files=False,
        save_histogram_files=False,
        names=None,
    )
    writer = writer_config.build_paired(
        experiment_dir=output_dir,
        n_initial_conditions=data.n_initial_conditions,
        n_timesteps=config_data["inference"]["n_forward_steps"],
        timestep=data.timestep,
        variable_metadata=variable_metadata,
        coords=data.coords,
        dataset_metadata=DatasetMetadata.from_env(),
    )

    logging.info("Running inference")
    record_logs = get_record_to_wandb(label="inference")
    run_inference(
        predict=stepper.predict_paired,
        data=data,
        aggregator=aggregator,
        writer=writer,
        record_logs=record_logs,
    )

    logging.info("Inference completed")
    writer.flush()
    aggregator.flush_diagnostics(subdir=f"epoch_{args.epoch:04d}")

    logging.info(f"Inference results saved to: {output_dir}")


if __name__ == "__main__":
    main()
