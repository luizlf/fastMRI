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

# Function to find optimal batch size directly
def find_optimal_batch_size(configs, experiment_name, max_batch_size, log_file, num_workers, 
                            challenge, precision, train_transform, val_transform, test_transform,
                            path_config, baseline_batch_size):
    """
    Find the optimal batch size without using separate processes
    """
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting batch size search for {experiment_name}\n")
    
    print(f"\nFinding optimal batch size for {experiment_name}...")
    
    # Start with a range of batch sizes that make sense for M4 Max
    # Try fewer batch sizes to reduce search time
    batch_sizes = [8, 16, 24, 32, 48]
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
    
    # Dictionary to store memory usage per batch size
    memory_usage = {}
    
    # Create a minimal dataset for testing batch sizes
    mini_data_module = FastMriDataModule(
        data_path=fetch_dir("knee_path", path_config),
        challenge=challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        sample_rate=0.05,  # Use only 5% of data for quick testing
        num_workers=num_workers,
        only_annotated=True
    )
    
    # Try each batch size directly without multiprocessing
    for batch_size in batch_sizes:
        try:
            print(f"  Testing batch size {batch_size}...")
            
            # Create a small model for testing
            test_model = UnetModule(
                in_chans=configs['in_chans'],
                out_chans=configs['out_chans'],
                chans=configs['chans'] // 2,  # Reduce channels for faster testing
                num_pool_layers=configs['num_pool_layers'],
                drop_prob=configs['drop_prob'],
                lr=configs['lr'],
                metric=configs['metric'],
                roi_weight=configs['roi_weight'],
                attn_layer=configs['attn_layer'],
                use_roi=configs['use_roi'],
                use_attention_gates=configs['use_attention_gates'],
            )
            
            # Override mini_data_module's batch size
            mini_data_module.batch_size = batch_size
            
            # Create simple trainer with minimal setup
            trainer = pl.Trainer(
                devices=configs['num_gpus'],
                max_epochs=1,
                accelerator=configs['backend'],
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=True,
                precision=precision,     # Use precision based on backend
                limit_train_batches=100,   # Only run 1 batch for testing
                limit_val_batches=20,     # Skip validation
                strategy="auto",
                use_distributed_sampler=False
            )
            
            
            # Prepare data to see if it fits in memory
            trainer.fit(test_model, datamodule=mini_data_module)
            
            # If we get here, the batch size works
            memory_usage[batch_size] = batch_size  # Use batch size as proxy for memory usage
            
            val_loss = trainer.callback_metrics.get("validation_loss", torch.tensor(float('inf'))).item()
            
            print(f"  Batch size {batch_size} completed successfully, validation loss = {val_loss:.20f}")
            
            # Clean up to free memory
            del trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  Batch size {batch_size} failed with error: {type(e).__name__}")
            print(f"  Error details: {str(e)}")
            # line where error occurred
            print(f"  Error occurred at line {e.__traceback__.tb_lineno}")
            # full traceback
            print(f"  Full traceback: {str(e.__traceback__)}")
            # break code execution
            raise e
            break
    
    # Find the largest batch size that worked
    optimal_batch_size = max(memory_usage.keys()) if memory_usage else baseline_batch_size
    
    with open(log_file, "a") as f:
        f.write(f"  Optimal batch size for {experiment_name}: {optimal_batch_size}\n")
    
    print(f"  Optimal batch size for {experiment_name}: {optimal_batch_size}")
    return optimal_batch_size

# Function to manually find a good learning rate
def find_optimal_learning_rate(configs, experiment_name, log_file, hyperparam_dir,
                               num_workers, challenge, precision, train_transform, val_transform, test_transform,
                               path_config):
    """
    Manual approach to find a good learning rate without spawning multiple processes
    """
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting learning rate search for {experiment_name}\n")
    
    print(f"\nFinding optimal learning rate for {experiment_name}...")
    
    # Define a range of learning rates to try
    lr_values = [0.0001, 0.0003, 0.001, 0.003, 0.01]
    
    # Sample a small subset of data for quick LR testing
    mini_data_module = FastMriDataModule(
        data_path=fetch_dir("knee_path", path_config),
        challenge=challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=configs['batch_size'],
        sample_rate=0.05,  # Use only 5% of data
        num_workers=num_workers,
        only_annotated=True
    )
    
    losses = []
    
    # Try each learning rate
    for lr in lr_values:
        # Create a model with current configuration and this learning rate
        model = UnetModule(
            in_chans=configs['in_chans'],
            out_chans=configs['out_chans'],
            chans=configs['chans'],
            num_pool_layers=configs['num_pool_layers'],
            drop_prob=configs['drop_prob'],
            lr=lr,  # Test this learning rate
            metric=configs['metric'],
            roi_weight=configs['roi_weight'],
            attn_layer=configs['attn_layer'],
            use_roi=configs['use_roi'],
            use_attention_gates=configs['use_attention_gates'],
        )
        
        # Create a trainer for quick LR testing
        trainer = pl.Trainer(
            devices=configs['num_gpus'],
            accelerator=configs['backend'],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            max_epochs=1,
            limit_train_batches=100,  # Run only 10 batches
            limit_val_batches=20,     # Run only 5 val batches
            precision=precision,     # Use precision based on backend
        )
        
        try:
            # Train for a short time
            trainer.fit(model, datamodule=mini_data_module)
            
            # Get the final validation loss
            val_loss = trainer.callback_metrics.get("validation_loss", torch.tensor(float('inf'))).item()
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
        plt.plot(lrs, loss_values, 'o-')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Loss')
        plt.title(f'Learning Rate Search for {experiment_name}')
        plt.grid(True)
        plt.savefig(hyperparam_dir / f"{experiment_name}_lr_search.png")
        plt.close()
        
        with open(log_file, "a") as f:
            f.write(f"  Best learning rate for {experiment_name}: {best_lr} (loss: {best_loss:.6f})\n")
        
        return best_lr
    else:
        print(f"  No valid learning rates found. Using default: {configs['lr']}")
        with open(log_file, "a") as f:
            f.write(f"  No valid learning rates found. Using default: {configs['lr']}\n")
        
        return configs['lr']

# Main execution function
def run_experiments():
    # Fix for multiprocessing on macOS
    # mp.set_start_method('spawn', force=True)

    # Set process priority higher for better performance
    try:
        os.nice(-10)  # Higher priority on Unix/Mac
    except:
        pass  # Skip if not available

    # Common parameters for all experiments
    path_config = pathlib.Path("fastmri_dirs.yaml")
    max_epochs = 15  # Set this to your desired number of epochs
    baseline_batch_size = 16  # Start with a higher baseline for M4 Max
    num_workers = 2  # Set to 1 to minimize multiprocessing issues but still have some parallelism
    challenge = "singlecoil"
    num_gpus = 1
    backend = "mps"  # Metal Performance Shaders for Apple Silicon

    # Check if we need to disable automatic mixed precision on MPS backend
    # MPS does not fully support AMP the same way as CUDA
    use_amp = True  # By default, use AMP on MPS
    if backend == "mps":
        precision = 32  # Use 32-bit precision on MPS
        #precision = "16-mixed"  # Use 16-bit precision on MPS
    else:
        precision = 16  # Use 16-bit precision on other backends (CUDA)

    # Configure all the experiments to run
    experiments = [
        {
            "name": "baseline",
            "description": "Baseline U-Net",
            "params": {
                "attn_layer": False,
                "metric": "l1",
                "use_roi": False,
                "use_attention_gates": False
            }
        },
        {
            "name": "roi_focus",
            "description": "U-Net with ROI focus",
            "params": {
                "attn_layer": False,
                "metric": "l1",
                "use_roi": True,
                "use_attention_gates": False
            }
        },
        {
            "name": "cbam",
            "description": "U-Net with CBAM",
            "params": {
                "attn_layer": True,
                "metric": "l1",
                "use_roi": False,
                "use_attention_gates": False
            }
        },
        {
            "name": "attention_gates",
            "description": "U-Net with Attention Gates",
            "params": {
                "attn_layer": False,
                "metric": "l1",
                "use_roi": True,
                "use_attention_gates": True
            }
        },
        {
            "name": "full_attention",
            "description": "U-Net with CBAM and Attention Gates",
            "params": {
                "attn_layer": True,
                "metric": "l1",
                "use_roi": True,
                "use_attention_gates": True
            }
        }
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
        f.write(f"\n\n==== STARTING NEW EXPERIMENT RUN {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")

    # Set global random seed for reproducibility
    pl.seed_everything(42)

    # Create transforms once (shared across experiments)
    mask = create_mask_for_mask_type("random", [0.08], [4])
    train_transform = T.UnetDataTransform(challenge, mask_func=mask, use_seed=False)
    val_transform = T.UnetDataTransform(challenge, mask_func=mask)
    test_transform = T.UnetDataTransform(challenge)

    # Run each experiment in sequence
    for exp_idx, experiment in enumerate(experiments):
        # Log experiment start
        with open(log_file, "a") as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting experiment {exp_idx+1}/{len(experiments)}: {experiment['name']} - {experiment['description']}\n")
            for param_name, param_value in experiment['params'].items():
                f.write(f"  {param_name}: {param_value}\n")
        
        print(f"\n{'='*80}")
        print(f"Starting experiment {exp_idx+1}/{len(experiments)}: {experiment['name']} - {experiment['description']}")
        print(f"{'='*80}")
        
        # Extract parameters for this experiment
        attn_layer = experiment['params']['attn_layer']
        metric = experiment['params']['metric']
        use_roi = experiment['params']['use_roi']
        use_attention_gates = experiment['params']['use_attention_gates']
        
        # Create experiment name
        exp_name = f"unet_{experiment['name']}"
        version_name = f"{exp_name}_{metric}" + \
                      ("_roi" if use_roi else "") + \
                      ("_attn" if attn_layer else "") + \
                      ("_agate" if use_attention_gates else "")
        
        # Experiment directory
        exp_dir = fetch_dir("log_path", path_config) / "unet" / version_name
        os.makedirs(exp_dir, exist_ok=True)
        
        # Set up TensorBoard logger
        tensorboard = pl_loggers.TensorBoardLogger(
            save_dir=str(fetch_dir("log_path", path_config)),
            name="unet",
            version=version_name
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
            lr_step_size=40,
            lr_gamma=0.1,
            weight_decay=0.0,
            max_epochs=max_epochs,
            metric=metric,
            roi_weight=0.5,
            attn_layer=attn_layer,
            use_roi=use_roi,
            use_attention_gates=use_attention_gates,
        )
        
        # Find optimal batch size for this configuration
        max_batch_size = 32  # Set a maximum batch size for testing
        optimal_batch_size = find_optimal_batch_size(configs, version_name, max_batch_size, log_file, 
                                                     num_workers, challenge, precision, train_transform,
                                                     val_transform, test_transform, path_config, baseline_batch_size)
        configs['batch_size'] = optimal_batch_size
        
        # Create a data module with the optimal batch size
        data_module = FastMriDataModule(
            data_path=fetch_dir("knee_path", path_config),
            challenge=challenge,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            test_path=None,
            batch_size=optimal_batch_size,
            num_workers=num_workers,
            only_annotated=True  # Use only annotated slices
        )
        
        # Find optimal learning rate for this configuration
        optimal_lr = find_optimal_learning_rate(configs, version_name, log_file, hyperparam_dir, num_workers, 
                                                challenge, precision, train_transform, val_transform,
                                                test_transform, path_config)
        configs['lr'] = optimal_lr
        
        # Log the optimized hyperparameters
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Optimized hyperparameters for {version_name}:\n")
            f.write(f"  Batch size: {optimal_batch_size}\n")
            f.write(f"  Learning rate: {optimal_lr}\n")
        
        print(f"Optimized hyperparameters for {version_name}:")
        print(f"  Batch size: {optimal_batch_size}")
        print(f"  Learning rate: {optimal_lr}")
        
        # Create model with optimized hyperparameters
        model = UnetModule(
            in_chans=configs['in_chans'],
            out_chans=configs['out_chans'],
            chans=configs['chans'],
            num_pool_layers=configs['num_pool_layers'],
            drop_prob=configs['drop_prob'],
            lr=configs['lr'],  # Optimized learning rate
            lr_step_size=configs['lr_step_size'],
            lr_gamma=configs['lr_gamma'],
            weight_decay=configs['weight_decay'],
            metric=configs['metric'],
            roi_weight=configs['roi_weight'],
            attn_layer=configs['attn_layer'],
            use_roi=configs['use_roi'],
            use_attention_gates=configs['use_attention_gates'],
        )
        
        # Set up callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=exp_dir / "checkpoints",
                monitor="validation_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                verbose=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            # Early stopping if no improvement for 5 epochs
            pl.callbacks.EarlyStopping(
                monitor="validation_loss",
                patience=5,
                mode="min",
                verbose=True
            )
        ]
        
        # Create trainer with performance optimizations
        trainer = pl.Trainer(
            devices=configs['num_gpus'],
            max_epochs=configs['max_epochs'],
            default_root_dir=configs['default_root_dir'],
            accelerator=configs['backend'],
            callbacks=callbacks,
            logger=tensorboard,
            precision=precision,  # Use precision based on backend
            check_val_every_n_epoch=2,  # Reduce validation frequency
            gradient_clip_val=1.0,  # Add gradient clipping for stability
            deterministic=False,  # Disable deterministic mode for speed
        )
        
        # Train model
        try:
            start_time = time.time()
            trainer.fit(model, datamodule=data_module)
            end_time = time.time()
            
            # Log experiment completion
            train_time_mins = (end_time - start_time) / 60
            best_val_loss = trainer.callback_metrics.get("validation_loss", torch.tensor(float('inf'))).item()
            
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed experiment: {experiment['name']}\n")
                f.write(f"  Training time: {train_time_mins:.2f} minutes\n")
                f.write(f"  Best validation loss: {best_val_loss:.6f}\n")
                f.write(f"  Saved model to: {exp_dir}/checkpoints\n")
            
            print(f"Completed experiment: {experiment['name']}")
            print(f"  Training time: {train_time_mins:.2f} minutes")
            print(f"  Best validation loss: {best_val_loss:.6f}")
            print(f"  Saved model to: {exp_dir}/checkpoints")
            
            # Optional: Run a test step with the best model
            if hasattr(trainer.checkpoint_callback, 'best_model_path') and trainer.checkpoint_callback.best_model_path:
                print(f"Loading best model from {trainer.checkpoint_callback.best_model_path}")
                best_model = UnetModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
                test_results = trainer.test(best_model, datamodule=data_module)
                
                with open(log_file, "a") as f:
                    f.write(f"  Test results: {test_results}\n")
                
                print(f"  Test results: {test_results}")
        
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR in experiment {experiment['name']}: {str(e)}\n")
            print(f"ERROR in experiment {experiment['name']}: {str(e)}")
            continue  # Continue to next experiment even if this one fails
        
        # Clear memory between experiments
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Add a small delay between experiments
        time.sleep(5)

    # Log completion of all experiments
    with open(log_file, "a") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All experiments completed!\n")

    print("\nAll experiments completed!")
    print(f"See log file at {log_file} for details.")

# Execute the proper way depending on environment
if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    run_experiments()
else:
    # In a notebook environment, still ensure proper method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass
    run_experiments()