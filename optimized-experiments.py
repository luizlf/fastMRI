"""
Optimized Sequential U-Net Experiments with Proper Hyperparameter Search
Fixed for running in Jupyter notebook while maintaining functionality
"""

import fastmri
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
import pytorch_lightning as pl
import os
import time
import math
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from src.subsample import create_mask_for_mask_type, RandomMaskFunc
from src import transforms as T
from src.mri_data import fetch_dir
from src.data_module import FastMriDataModule, AnnotatedSliceDataset
from src.unet.unet_module import UnetModule
from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.tuner.tuning import Tuner
import multiprocessing as mp
import warnings

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    message="The .*dataloader' does not have many workers.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
    category=UserWarning,
)


# Function to find optimal batch size directly
def find_optimal_batch_size(
    configs,
    experiment_name,
    max_batch_size,
    log_file,
    num_workers,
    challenge,
    precision,
    train_transform,
    val_transform,
    test_transform,
    path_config,
    baseline_batch_size,
):
    """
    Find the optimal batch size without using separate processes,
    processing roughly the same number of images for each test.
    """
    with open(log_file, "a") as f:
        f.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting batch size search for {experiment_name}\n"
        )

    print(f"\nFinding optimal batch size for {experiment_name}...")

    # <<< Define target number of images for fair comparison >>>
    target_images_per_test = 192  # Adjust as needed

    # Start with a range of batch sizes that make sense for M4 Max
    batch_sizes = [2, 4, 8, 16]
    batch_sizes = sorted([bs for bs in batch_sizes if bs <= max_batch_size])

    # Dictionary to store timing results per batch size
    timing_results = {}

    # Create a minimal dataset for testing batch sizes
    mini_data_module = FastMriDataModule(
        data_path=fetch_dir("knee_path", path_config),
        challenge=challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        sample_rate=0.05,  # Use only 5% of data for quick testing
        num_workers=num_workers,
        only_annotated=True,
        distributed_sampler=False,
    )

    # Try each batch size directly without multiprocessing
    for batch_size in batch_sizes:
        try:
            print(f"  Testing batch size {batch_size}...")

            # <<< Calculate number of batches dynamically >>>
            num_batches_train = math.ceil(target_images_per_test / batch_size)
            num_batches_val = math.ceil(
                num_batches_train / 4
            )  # Roughly 1/4th val batches
            print(
                f"    Target images: {target_images_per_test}, Train batches: {num_batches_train}, Val batches: {num_batches_val}"
            )

            # Create a small model for testing
            test_model = UnetModule(
                in_chans=configs["in_chans"],
                out_chans=configs["out_chans"],
                chans=configs["chans"],
                num_pool_layers=configs["num_pool_layers"],
                drop_prob=configs["drop_prob"],
                lr=configs["lr"],
                metric=configs["metric"],
                roi_weight=configs["roi_weight"],
                attn_layer=configs["attn_layer"],
                use_roi=configs["use_roi"],
                use_attention_gates=configs["use_attention_gates"],
            )

            # Override mini_data_module's batch size
            mini_data_module.batch_size = batch_size

            # Create simple trainer with minimal setup
            trainer = pl.Trainer(
                devices=configs["num_gpus"],
                max_epochs=1,
                accelerator=configs["backend"],
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=True,
                precision=precision,
                # <<< Use dynamic batch limits >>>
                limit_train_batches=num_batches_train,
                limit_val_batches=num_batches_val,
                strategy="auto",
                use_distributed_sampler=False,
            )

            # Prepare data and time the fit process
            # <<< Remove nested try/except for KeyboardInterrupt >>>
            # try:
            if __name__ == "__main__":
                print(f"  Starting batch size test for batch size {batch_size}")
                start_time = time.time()
                trainer.fit(test_model, datamodule=mini_data_module)
                end_time = time.time()
                duration = end_time - start_time

                # If we get here, the batch size works
                timing_results[batch_size] = duration

                val_loss = trainer.callback_metrics.get(
                    "val_loss", torch.tensor(float("inf"))
                ).item()

                print(
                    f"  Batch size {batch_size} completed successfully in {duration:.2f} seconds, validation loss = {val_loss:.6f}"
                )

            # except KeyboardInterrupt:
            #     print(f"\n  >>> User interrupted batch size test at batch size {batch_size}. <<<\n")
            #     # <<< Remove explicit stop >>>
            #     # trainer.should_stop = True
            #     # Break the loop testing batch sizes
            #     break
            # <<< End nested try/except >>>

            # Clean up to free memory (moved back inside main try block logic)
            del trainer, test_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Outer exception handling for other errors (like OOM)
        except Exception as e:
            print(f"  Batch size {batch_size} failed with error: {type(e).__name__}")
            print(f"  Error details: {str(e)}")
            # ... (traceback printing) ...
            # Break loop on other errors too
            break

    # Find the batch size with the fastest execution time
    if timing_results:
        optimal_batch_size = min(timing_results, key=timing_results.get)
        fastest_time = timing_results[optimal_batch_size]
        print(
            f"  Fastest time ({fastest_time:.2f}s) achieved with batch size {optimal_batch_size}."
        )
    else:
        optimal_batch_size = baseline_batch_size
        print(
            f"  No batch sizes completed successfully. Falling back to baseline: {baseline_batch_size}"
        )

    with open(log_file, "a") as f:
        f.write(
            f"  Optimal batch size for {experiment_name}: {optimal_batch_size} (based on speed)\n"
        )

    print(f"  Optimal batch size for {experiment_name}: {optimal_batch_size}")
    return optimal_batch_size


# Function to manually find a good learning rate
def find_optimal_learning_rate(
    configs,
    experiment_name,
    log_file,
    hyperparam_dir,
    num_workers,
    challenge,
    precision,
    train_transform,
    val_transform,
    test_transform,
    path_config,
):
    """
    Manual approach to find a good learning rate without spawning multiple processes
    """
    with open(log_file, "a") as f:
        f.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting learning rate search for {experiment_name}\n"
        )

    print(f"\nFinding optimal learning rate for {experiment_name}...")

    # Define a range of learning rates to try
    lr_values = [0.0001, 0.0003, 0.001, 0.003, 0.01]

    # Create a minimal dataset for testing batch sizes
    mini_data_module = FastMriDataModule(
        data_path=fetch_dir("knee_path", path_config),
        challenge=challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=configs["batch_size"],
        sample_rate=0.05,  # Use only 5% of data
        num_workers=num_workers,
        only_annotated=True,
    )

    losses = []

    # Try each learning rate
    for lr in lr_values:
        # Create a model with current configuration and this learning rate
        model = UnetModule(
            in_chans=configs["in_chans"],
            out_chans=configs["out_chans"],
            chans=configs["chans"],
            num_pool_layers=configs["num_pool_layers"],
            drop_prob=configs["drop_prob"],
            lr=lr,  # Test this learning rate
            metric=configs["metric"],
            roi_weight=configs["roi_weight"],
            attn_layer=configs["attn_layer"],
            use_roi=configs["use_roi"],
            use_attention_gates=configs["use_attention_gates"],
        )

        # Create a trainer for quick LR testing
        trainer = pl.Trainer(
            devices=configs["num_gpus"],
            accelerator=configs["backend"],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            max_epochs=1,
            limit_train_batches=50,  # Run only 10 batches
            limit_val_batches=20,  # Run only 5 val batches
            precision=precision,  # Use precision based on backend
        )

        try:
            # Train for a short time
            if __name__ == "__main__":
                print(f"  Starting learning rate test for learning rate {lr}")
                trainer.fit(model, datamodule=mini_data_module)

            # Get the final validation loss
            val_loss = trainer.callback_metrics.get(
                "val_loss", torch.tensor(float("inf"))
            ).item()
            losses.append((lr, val_loss))

            print(f"  Learning rate {lr}: validation loss = {val_loss:.20f}")

        except Exception as e:
            print(f"  Learning rate {lr} failed with error: {type(e).__name__}")
            print(f"  Error details: {str(e)}")
            print(f"  Error occurred at line {e.__traceback__.tb_lineno}")
            print(f"  Full traceback: {str(e.__traceback__)}")
            raise e
            # Continue to the next learning rate

        # Clean up
        del trainer, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Find the learning rate with the lowest validation loss
    if losses:
        best_lr, best_loss = min(losses, key=lambda x: x[1])
        print(f"  Best learning rate: {best_lr} (loss: {best_loss:.6f})")

        # Plot learning rate vs loss
        plt.figure(figsize=(10, 6))
        lrs, loss_values = zip(*losses)
        plt.plot(lrs, loss_values, "o-")
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Validation Loss")
        plt.title(f"Learning Rate Search for {experiment_name}")
        plt.grid(True)
        plt.savefig(hyperparam_dir / f"{experiment_name}_lr_search.png")
        plt.close()

        with open(log_file, "a") as f:
            f.write(
                f"  Best learning rate for {experiment_name}: {best_lr} (loss: {best_loss:.6f})\n"
            )

        return best_lr
    else:
        print(f"  No valid learning rates found. Using default: {configs['lr']}")
        with open(log_file, "a") as f:
            f.write(
                f"  No valid learning rates found. Using default: {configs['lr']}\n"
            )

        return configs["lr"]


# Main execution function
def run_experiments():
    # Fix for multiprocessing on macOS
    # mp.set_start_method('spawn', force=True)

    # Set process priority higher for better performance
    try:
        os.nice(-10)  # Higher priority on Unix/Mac
    except:
        pass  # Skip if not available

    # Dictionary to store previously optimized hyperparameters
    # Extracted from experiment_log.txt as of last successful run in provided log
    OPTIMIZED_PARAMS = {
        "unet_baseline_l1": {"batch_size": 8, "lr": 0.0001},
        "unet_roi_focus_l1_roi": {"batch_size": 8, "lr": 0.0001},
        "unet_cbam_l1_attn": {"batch_size": 8, "lr": 0.0001},
        "unet_full_attention_l1_roi_attn_agate": {"batch_size": 8, "lr": 0.0001},
        "unet_attention_gates_l1_roi_agate": {"batch_size": 8, "lr": 0.0001},
        "unet_full_attention_no_roi_l1_attn_agate": {"batch_size": 8, "lr": 0.0001},
        "unet_attention_gates_no_roi_l1_agate": {"batch_size": 8, "lr": 0.0001},
        "unet_baseline_l1_ssim": {"batch_size": 8, "lr": 0.0001},
        "unet_full_attention_l1_ssim_roi_attn_agate": {"batch_size": 8, "lr": 0.0001},
        "unet_attention_gates_l1_ssim_roi_agate": {"batch_size": 8, "lr": 0.0001},
        "unet_full_attention_l1_ssim_l1_ssim_attn_agate": {
            "batch_size": 8,
            "lr": 0.0001,
        },
        "unet_attention_gates_no_roi_l1_ssim_agate": {"batch_size": 8, "lr": 0.0001},
        "unet_full_attention_l1_ssim_roi_l1_ssim_roi_attn_agate": {
            "batch_size": 8,
            "lr": 0.0001,
        },
        # Add entries for other experiments as they are successfully run
        # "unet_attention_gates_l1_roi_agate": { ... },
        # "unet_full_attention_l1_roi_attn_agate": { ... },
    }

    # Ask user whether to use stored params or rerun search
    # <<< Default back to 'y' for full run >>>
    use_stored_params_input = (
        input("Use stored optimal batch size and learning rate? (y/n, default: y): ")
        .lower()
        .strip()
    )
    USE_STORED_PARAMS = not (
        use_stored_params_input == "n" or use_stored_params_input == "no"
    )

    if USE_STORED_PARAMS:
        print(
            "\n>>> Attempting to use stored hyperparameters. <<<"
            + " If not found, search will run."
        )  # Added clarification
    else:
        print(
            "\n>>> Rerunning hyperparameter search for all selected experiments. <<<"
            + " (Set default input to 'y' to skip)"
        )

    # Common parameters for all experiments
    path_config = pathlib.Path("fastmri_dirs.yaml")
    # <<< Increase max_epochs for full run >>>
    max_epochs = 30  # Set this to your desired number of epochs
    baseline_batch_size = 8  # Keep lower baseline from testing
    num_workers = 0  # Keep num_workers=0 for stability based on previous tests
    challenge = "singlecoil"
    num_gpus = 1
    backend = "mps"  # Metal Performance Shaders for Apple Silicon

    # Check if we need to disable automatic mixed precision on MPS backend
    # MPS does not fully support AMP the same way as CUDA
    use_amp = True  # By default, use AMP on MPS
    if backend == "mps":
        # precision = 32  # Use 32-bit precision on MPS
        precision = "16-mixed"  # Use 16-bit precision on MPS
    else:
        precision = 16  # Use 16-bit precision on other backends (CUDA)

    # Configure all the experiments to run
    experiments = [
        {
            "name": "baseline",
            "description": "Baseline U-Net, No ROI, No Attention Gates",
            "params": {
                "attn_layer": False,
                "metric": "l1",
                "use_roi": False,
                "use_attention_gates": False,
            },
        },
        {
            "name": "baseline_roi",
            "description": "Baseline U-Net with ROI focus",
            "params": {
                "attn_layer": False,
                "metric": "l1",
                "use_roi": True,
                "use_attention_gates": False,
            },
        },
        {
            "name": "cbam",
            "description": "U-Net with CBAM",
            "params": {
                "attn_layer": True,
                "metric": "l1",
                "use_roi": False,
                "use_attention_gates": False,
            },
        },
        {
            "name": "cbam_roi",
            "description": "U-Net with CBAM and ROI",
            "params": {
                "attn_layer": True,
                "metric": "l1",
                "use_roi": True,
                "use_attention_gates": False,
            },
        },
        {
            "name": "attention_gates_roi",
            "description": "U-Net with Attention Gates and ROI",
            "params": {
                "attn_layer": False,
                "metric": "l1",
                "use_roi": True,
                "use_attention_gates": True,
            },
        },
        {
            "name": "full_attention_roi",
            "description": "U-Net with CBAM and Attention Gates and ROI",
            "params": {
                "attn_layer": True,
                "metric": "l1",
                "use_roi": True,
                "use_attention_gates": True,
            },
        },
        # <<< Add new configurations without ROI >>>
        {
            "name": "attention_gates",
            "description": "U-Net with Attention Gates (No ROI Loss)",
            "params": {
                "attn_layer": False,
                "metric": "l1",
                "use_roi": False,  # Explicitly False
                "use_attention_gates": True,
            },
        },
        {
            "name": "full_attention",
            "description": "U-Net with CBAM and Attention Gates (No ROI Loss)",
            "params": {
                "attn_layer": True,
                "metric": "l1",
                "use_roi": False,  # Explicitly False
                "use_attention_gates": True,
            },
        },
        {
            "name": "baseline_l1_ssim",
            "description": "Baseline U-Net (L1+SSIM Loss), No ROI",
            "params": {
                "attn_layer": False,
                "metric": "l1_ssim",
                "use_roi": False,
                "use_attention_gates": False,
            },
        },
        {
            "name": "baseline_l1_ssim_roi",
            "description": "Baseline U-Net (L1+SSIM Loss), with ROI",
            "params": {
                "attn_layer": False,
                "metric": "l1_ssim",
                "use_roi": True,
                "use_attention_gates": False,
            },
        },
        {
            "name": "full_attention_l1_ssim",
            "description": "U-Net with CBAM & Attn Gates (L1+SSIM Loss, No ROI)",
            "params": {
                "attn_layer": True,
                "metric": "l1_ssim",
                "use_roi": False,
                "use_attention_gates": True,
            },
        },
        {
            "name": "full_attention_l1_ssim_roi",
            "description": "U-Net with CBAM & Attn Gates (L1+SSIM Loss, with ROI)",
            "params": {
                "attn_layer": True,
                "metric": "l1_ssim",
                "use_roi": True,
                "use_attention_gates": True,
            },
        },
        {
            "name": "cbam_l1_ssim",
            "description": "U-Net with CBAM and L1+SSIM Loss",
            "params": {
                "attn_layer": True,
                "metric": "l1_ssim",
                "use_roi": False,
                "use_attention_gates": False,
            },
        },
        {
            "name": "cbam_l1_ssim_roi",
            "description": "U-Net with CBAM and L1+SSIM Loss and ROI",
            "params": {
                "attn_layer": True,
                "metric": "l1_ssim",
                "use_roi": True,
                "use_attention_gates": False,
            },
        },
    ]

    # Create results directory
    results_dir = pathlib.Path("./experiment_results")
    results_dir.mkdir(exist_ok=True)

    # Create a hyperparameter search directory
    hyperparam_dir = results_dir / "hyperparameter_search"
    hyperparam_dir.mkdir(exist_ok=True)

    # Log file to track progress
    log_file = results_dir / "experiment_log.txt"
    with open(log_file, "a") as f:
        f.write(
            f"\n\n==== STARTING NEW EXPERIMENT RUN {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n"
        )

    # Set global random seed for reproducibility
    pl.seed_everything(42)

    # Create transforms once (shared across experiments)
    mask = create_mask_for_mask_type("random", [0.08], [4])
    train_transform = T.UnetDataTransform(challenge, mask_func=mask, use_seed=False)
    val_transform = T.UnetDataTransform(challenge, mask_func=mask)
    test_transform = T.UnetDataTransform(challenge)

    # Run each experiment in sequence
    # <<< Include all experiments for full run >>>
    target_experiment_names = [
        # "attention_gates_no_roi",
        # "full_attention_no_roi",
        # "baseline_l1_ssim",
        # "cbam_l1_ssim",
        "attention_gates_l1_ssim",
        "full_attention_l1_ssim",
        "baseline_l1_ssim_roi",
        "cbam_l1_ssim_roi",
        "attention_gates_l1_ssim_roi",
        "full_attention_l1_ssim_roi",
        # "full_attention_l1_ssim_roi",
        # "baseline",
        # "cbam",
        # "full_attention",
        # "roi_focus",
        # "attention_gates",
        # "full_attention",
    ]  # Add other names if needed, e.g., "full_attention"

    # <<< Keep ordered execution >>>
    experiments_dict = {exp["name"]: exp for exp in experiments}
    experiments_to_run = [
        experiments_dict[name]
        for name in target_experiment_names
        if name in experiments_dict
    ]

    if not experiments_to_run:
        print(f"ERROR: No experiments found with names: {target_experiment_names}")
        return

    print(
        f"\n>>> Running specified experiments in order: {target_experiment_names} <<<\n"
    )

    for exp_idx, experiment in enumerate(experiments_to_run):
        # Log experiment start
        with open(log_file, "a") as f:
            f.write(
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting experiment {exp_idx+1}/{len(experiments_to_run)}: {experiment['name']} - {experiment['description']}\n"
            )
            for param_name, param_value in experiment["params"].items():
                f.write(f"  {param_name}: {param_value}\n")

        print(f"\n{'='*80}")
        print(
            f"Starting experiment {exp_idx+1}/{len(experiments_to_run)}: {experiment['name']} - {experiment['description']}"
        )
        print(f"{'='*80}")

        # Extract parameters for this experiment
        attn_layer = experiment["params"]["attn_layer"]
        metric = experiment["params"]["metric"]
        use_roi = experiment["params"]["use_roi"]
        use_attention_gates = experiment["params"]["use_attention_gates"]

        # Create experiment name
        exp_name = f"unet_{experiment['name']}"
        version_name = (
            f"{exp_name}_{metric}"
            + ("_roi" if use_roi else "")
            + ("_attn" if attn_layer else "")
            + ("_agate" if use_attention_gates else "")
        )

        # Experiment directory
        exp_dir = fetch_dir("log_path", path_config) / "unet" / version_name
        os.makedirs(exp_dir, exist_ok=True)

        # Set up TensorBoard logger
        tensorboard = pl_loggers.TensorBoardLogger(
            save_dir=str(fetch_dir("log_path", path_config)),
            name="unet",
            version=version_name,
        )

        # Create model configuration
        configs = dict(
            challenge=challenge,
            num_gpus=num_gpus,
            backend=backend,
            batch_size=baseline_batch_size,  # Will be updated after search
            data_path=fetch_dir("knee_path", path_config),
            default_root_dir=exp_dir,
            mode="train",
            mask_type="random",
            center_fractions=[0.08],
            accelerations=[4],
            in_chans=1,
            out_chans=1,
            chans=64,
            num_pool_layers=3,
            drop_prob=0.0,
            lr=0.001,  # Default, will be updated after search
            lr_factor=0.1,  # Default ReduceLROnPlateau factor
            lr_patience=2,  # Default ReduceLROnPlateau patience
            weight_decay=0.01,
            max_epochs=max_epochs,
            metric=metric,
            roi_weight=0.5,
            l1_ssim_alpha=0.7,
            attn_layer=attn_layer,
            use_roi=use_roi,
            use_attention_gates=use_attention_gates,
        )

        # --- Start of Modified Section ---
        run_hyperparam_search = True  # Default to running search
        if USE_STORED_PARAMS and version_name in OPTIMIZED_PARAMS:
            print(f"  Found stored parameters for {version_name}.")
            stored_values = OPTIMIZED_PARAMS[version_name]
            optimal_batch_size = stored_values["batch_size"]
            optimal_lr = stored_values["lr"]
            configs["batch_size"] = optimal_batch_size
            configs["lr"] = optimal_lr
            run_hyperparam_search = False  # Skip search
            # Log the usage of stored parameters
            with open(log_file, "a") as f:
                f.write(
                    f"  Using stored hyperparameters: Batch Size={optimal_batch_size}, LR={optimal_lr}\n"
                )
        elif USE_STORED_PARAMS:
            print(f"  Stored parameters not found for {version_name}. Running search.")
        # --- End of Modified Section ---

        if run_hyperparam_search:
            print(f"  Running hyperparameter search for {version_name}...")
            # Find optimal batch size
            max_batch_size = 32  # Set a maximum batch size for testing
            optimal_batch_size = find_optimal_batch_size(
                configs,
                version_name,
                max_batch_size,
                log_file,
                num_workers,
                challenge,
                precision,
                train_transform,
                val_transform,
                test_transform,
                path_config,
                baseline_batch_size,
            )
            configs["batch_size"] = optimal_batch_size

            # Find optimal learning rate
            optimal_lr = find_optimal_learning_rate(
                configs,
                version_name,
                log_file,
                hyperparam_dir,
                num_workers,
                challenge,
                precision,
                train_transform,
                val_transform,
                test_transform,
                path_config,
            )
            configs["lr"] = optimal_lr
        # <---- End of the if run_hyperparam_search block

        # Log the final hyperparameters used (either stored or found)
        with open(log_file, "a") as f:
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final hyperparameters for {version_name}:\n"
            )
            f.write(f"  Batch size: {configs['batch_size']}\n")
            f.write(f"  Learning rate: {configs['lr']}\n")

        print(f"Final hyperparameters for {version_name}:")
        print(f"  Batch size: {configs['batch_size']}")
        print(f"  Learning rate: {configs['lr']}")

        # Create data module with the final batch size
        print(
            f"  Creating DataModule with batch size: {configs['batch_size']}"
        )  # Added print for verification
        data_module = FastMriDataModule(
            data_path=fetch_dir("knee_path", path_config),
            challenge=challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            test_path=None,
            batch_size=configs["batch_size"],  # Use final batch size
            num_workers=num_workers,
            only_annotated=True,
        )

        # Create model with final hyperparameters
        print(
            f"  Creating Model with LR: {configs['lr']}"
        )  # Added print for verification
        model = UnetModule(
            in_chans=configs["in_chans"],
            out_chans=configs["out_chans"],
            chans=configs["chans"],
            num_pool_layers=configs["num_pool_layers"],
            drop_prob=configs["drop_prob"],
            lr=configs["lr"],  # Use final LR
            lr_factor=configs["lr_factor"],
            lr_patience=configs["lr_patience"],
            weight_decay=configs["weight_decay"],
            metric=configs["metric"],
            roi_weight=configs["roi_weight"],
            attn_layer=configs["attn_layer"],
            use_roi=configs["use_roi"],
            use_attention_gates=configs["use_attention_gates"],
            l1_ssim_alpha=configs["l1_ssim_alpha"],
        )

        # Set up callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=exp_dir / "checkpoints",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                verbose=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            # Early stopping if no improvement for 5 epochs
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, mode="min", verbose=True
            ),
        ]

        # Create trainer with performance optimizations
        # <<< Remove batch limits for full run >>>
        trainer = pl.Trainer(
            devices=configs["num_gpus"],
            max_epochs=configs["max_epochs"],  # Use updated max_epochs
            default_root_dir=configs["default_root_dir"],
            accelerator=configs["backend"],
            callbacks=callbacks,
            logger=tensorboard,
            precision=precision,  # Use precision based on backend
            check_val_every_n_epoch=1,  # Run validation every epoch
            gradient_clip_val=1.0,  # Add gradient clipping for stability
            deterministic=False,  # Disable deterministic mode for speed
            # limit_train_batches=10, # REMOVED
            # limit_val_batches=5    # REMOVED
        )

        # Train model
        try:
            start_time = time.time()
            print(f"  Starting full training for {version_name}...")  # Updated print
            trainer.fit(model, datamodule=data_module)
            end_time = time.time()

            # Log experiment completion
            train_time_mins = (end_time - start_time) / 60
            # <<< Get metrics from trainer state >>>
            final_metrics = trainer.callback_metrics
            best_val_loss = final_metrics.get(
                "val_loss", torch.tensor(float("inf"))
            ).item()
            last_val_image_loss = final_metrics.get(
                "val_loss_image", torch.tensor(float("nan"))
            ).item()
            last_val_roi_loss = final_metrics.get(
                "val_loss_roi", torch.tensor(float("nan"))
            ).item()

            with open(log_file, "a") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed experiment: {experiment['name']}\n"
                )
                f.write(f"  Training time: {train_time_mins:.2f} minutes\n")
                f.write(
                    f"  Best overall validation loss (callback monitor): {best_val_loss:.6f}\n"
                )
                # <<< Log final epoch losses >>>
                f.write(
                    f"  Final Epoch Validation Image Loss: {last_val_image_loss:.6f}\n"
                )
                f.write(f"  Final Epoch Validation ROI Loss: {last_val_roi_loss:.6f}\n")
                f.write(f"  Saved model to: {exp_dir}/checkpoints\n")

            print(f"Completed experiment: {experiment['name']}")
            print(f"  Training time: {train_time_mins:.2f} minutes")
            print(
                f"  Best overall validation loss (callback monitor): {best_val_loss:.6f}"
            )
            # <<< Print final epoch losses >>>
            print(f"  Final Epoch Validation Image Loss: {last_val_image_loss:.6f}")
            print(f"  Final Epoch Validation ROI Loss: {last_val_roi_loss:.6f}")
            print(f"  Saved model to: {exp_dir}/checkpoints")

            # Optional: Run a test step with the best model
            if (
                hasattr(trainer.checkpoint_callback, "best_model_path")
                and trainer.checkpoint_callback.best_model_path
            ):
                print(
                    f"Loading best model from {trainer.checkpoint_callback.best_model_path}"
                )
                best_model = UnetModule.load_from_checkpoint(
                    trainer.checkpoint_callback.best_model_path
                )
                test_results = trainer.test(best_model, datamodule=data_module)

                with open(log_file, "a") as f:
                    f.write(f"  Test results: {test_results}\n")

                print(f"  Test results: {test_results}")

        except Exception as e:
            with open(log_file, "a") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR in experiment {experiment['name']}: {str(e)}\n"
                )
            print(f"ERROR in experiment {experiment['name']}: {str(e)}")
            continue  # Continue to next experiment even if this one fails

        # Clear memory between experiments
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Add a small delay between experiments
        time.sleep(5)

    # Log completion of all experiments
    with open(log_file, "a") as f:
        f.write(
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All experiments completed!\n"
        )

    print("\nAll experiments completed!")
    print(f"See log file at {log_file} for details.")


# Execute the proper way depending on environment
if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    run_experiments()
else:
    # In a notebook environment, still ensure proper method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Method already set
        pass
    run_experiments()
