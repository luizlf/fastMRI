import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch_lightning as pl

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.unet.unet_module import UnetModule
    from src.data_module import FastMriDataModule
    from src import transforms as T
    from src.subsample import create_mask_for_mask_type
    import torchvision
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(
        f"Ensure the script is run from the project root or 'src' is in the Python path."
    )
    print(f"Project root added to path: {project_root}")
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_dir(key: str, data_config_file: str | Path) -> Path:
    """Fetches directory path from the config file."""
    # Simplified version for this script
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        raise FileNotFoundError(f"Path config file not found: {data_config_file}")
    with open(data_config_file, "r") as f:
        config = yaml.safe_load(f)
        data_dir = config.get(key)
        if data_dir is None:
            raise KeyError(f"Key '{key}' not found in {data_config_file}")
    path = Path(data_dir)
    if not path.is_dir():
        logging.warning(f"Directory specified by key '{key}' does not exist: {path}")
    return path


def visualize_checkpoints(args):
    """Loads checkpoints, runs validation, and logs images to TensorBoard."""

    base_checkpoint_dir = Path(args.checkpoint_dir)
    # Remove separate visualization log directory
    # vis_log_dir = Path(args.vis_log_dir)
    # vis_log_dir.mkdir(parents=True, exist_ok=True)
    # logging.info(f"Visualization logs will be saved to: {vis_log_dir}")
    logging.info(
        f"Visualization logs will be added to existing experiment directories."
    )

    # --- 1. Find Checkpoints ---
    checkpoints_to_process = []
    if not base_checkpoint_dir.is_dir():
        logging.error(f"Checkpoint directory not found: {base_checkpoint_dir}")
        return

    logging.info(f"Scanning for checkpoints in {base_checkpoint_dir}...")
    # Look for any subdirectory that might contain checkpoints
    for version_dir in sorted(base_checkpoint_dir.iterdir()):
        if version_dir.is_dir():
            checkpoint_subdir = version_dir / "checkpoints"
            if checkpoint_subdir.is_dir():
                # Prioritize last.ckpt, then look for any .ckpt
                ckpt_path = checkpoint_subdir / "last.ckpt"
                if not ckpt_path.is_file():
                    found_ckpts = list(checkpoint_subdir.glob("*.ckpt"))
                    if found_ckpts:
                        # Maybe sort by modification time or name if needed
                        ckpt_path = sorted(found_ckpts)[
                            -1
                        ]  # Take the last one alphabetically
                        logging.info(
                            f"  'last.ckpt' not found in {version_dir.name}, using {ckpt_path.name}"
                        )
                    else:
                        logging.warning(
                            f"  No .ckpt files found in {checkpoint_subdir}"
                        )
                        continue  # Skip this version if no checkpoint found

                if ckpt_path.is_file():
                    checkpoints_to_process.append(
                        {"path": ckpt_path, "version_name": version_dir.name}
                    )
                    logging.info(
                        f"  Found checkpoint for {version_dir.name}: {ckpt_path.name}"
                    )

    if not checkpoints_to_process:
        logging.error(f"No valid checkpoints found in {base_checkpoint_dir}")
        return
    logging.info(f"Found {len(checkpoints_to_process)} checkpoints to visualize.")

    # --- 2. Setup Data ---
    logging.info("Setting up data module and transforms...")
    try:
        data_path = fetch_dir(f"{args.subsplit}_path", args.data_config)
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Error fetching data path: {e}")
        return

    # Create transforms (use fixed settings matching typical validation)
    # We don't need a mask_func for validation typically, but UnetDataTransform might expect it.
    # Using a dummy mask or ensuring the transform handles None is needed.
    # Let's assume validation doesn't usually use acceleration mask.
    val_transform = T.UnetDataTransform(which_challenge=args.challenge, mask_func=None)

    data_module = FastMriDataModule(
        data_path=data_path,
        challenge=args.challenge,
        train_transform=None,  # Not needed
        val_transform=val_transform,
        test_transform=None,  # Not needed
        test_split="val",  # We are using the val split
        sample_rate=1.0,  # Use full validation set
        batch_size=args.batch_size,
        num_workers=0,  # Important for stability in this script
        distributed_sampler=False,
        only_annotated=True,  # Match training setup if applicable
    )
    try:
        val_dataloader = data_module.val_dataloader()
        logging.info("Validation DataLoader created.")
    except Exception as e:
        logging.error(f"Failed to create DataLoader: {e}")
        return

    # --- 3. Determine Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # --- 4. Loop Through Checkpoints and Visualize ---
    for ckpt_info in checkpoints_to_process:
        ckpt_path = ckpt_info["path"]
        version_name = ckpt_info["version_name"]
        # <<< Use the original version directory for logging >>>
        writer_log_dir = base_checkpoint_dir / version_name
        logging.info(f"\nProcessing {version_name} from {ckpt_path.name}...")

        # Create SummaryWriter for this specific checkpoint, saving in its original log dir
        # writer_log_dir = vis_log_dir / version_name
        try:
            # SummaryWriter will append to existing event files in this directory
            writer = SummaryWriter(log_dir=str(writer_log_dir))
            logging.info(f"  TensorBoard writer logging to: {writer_log_dir}")
        except Exception as e:
            logging.error(
                f"  Failed to create SummaryWriter for {version_name} in {writer_log_dir}: {e}"
            )
            continue  # Skip to next checkpoint

        try:
            # Load model
            model = UnetModule.load_from_checkpoint(
                checkpoint_path=str(ckpt_path), map_location=device
            )
            model.eval()  # Set to evaluation mode
            model.to(device)
            logging.info(f"  Model loaded successfully to {device}.")

            # Iterate validation batches
            batch_count = 0
            with torch.no_grad():  # Disable gradient calculations
                with tqdm(
                    total=(
                        min(len(val_dataloader), args.num_batches)
                        if args.num_batches > 0
                        else len(val_dataloader)
                    ),
                    desc=f"  Validating {version_name}",
                ) as pbar:
                    for batch_idx, batch in enumerate(val_dataloader):
                        if args.num_batches > 0 and batch_idx >= args.num_batches:
                            break

                        # Move batch to device - Assign to NEW variables
                        try:
                            image_dev = batch.image.to(device)
                            target_dev = batch.target.to(device)
                            mean_dev = batch.mean.to(device)
                            std_dev = batch.std.to(device)
                            # annotations_dev = batch.annotations # Keep annotations on CPU or handle differently if needed
                        except AttributeError as e:
                            logging.error(
                                f"  Error accessing expected attributes in batch {batch_idx}. Skipping batch. Error: {e}"
                            )
                            continue

                        # Forward pass (use device tensors)
                        output = model(image_dev)

                        # --- Prepare images for logging ---
                        # Detach, move to CPU, denormalize (use device tensors for mean/std)
                        mean_unsqueezed = mean_dev.unsqueeze(1).unsqueeze(2)
                        std_unsqueezed = std_dev.unsqueeze(1).unsqueeze(2)
                        output_viz = (output * std_unsqueezed + mean_unsqueezed).cpu()
                        # Use target_dev here for consistency
                        target_viz = (
                            target_dev * std_unsqueezed + mean_unsqueezed
                        ).cpu()

                        # Add channel dimension if missing (output should have it)
                        if output_viz.ndim == 3:
                            output_viz = output_viz.unsqueeze(1)
                        if target_viz.ndim == 3:
                            target_viz = target_viz.unsqueeze(1)

                        # Normalize/Clamp to [0, 1] for visualization
                        out_min, out_max = output_viz.min(), output_viz.max()
                        tgt_min, tgt_max = target_viz.min(), target_viz.max()
                        if out_max > out_min:
                            output_viz = (output_viz - out_min) / (out_max - out_min)
                        if tgt_max > tgt_min:
                            target_viz = (target_viz - tgt_min) / (tgt_max - tgt_min)
                        output_viz = torch.clamp(output_viz, 0.0, 1.0)
                        target_viz = torch.clamp(target_viz, 0.0, 1.0)

                        diff_viz = torch.abs(output_viz - target_viz)

                        # --- Log each image individually with detailed tags ---
                        batch_size = output_viz.shape[0]
                        for i in range(batch_size):
                            try:
                                # Get filename and slice number for tagging
                                fname = Path(
                                    batch.fname[i]
                                ).stem  # Get filename without extension
                                slice_num = int(batch.slice_num[i].item())
                                # Use a similar tag structure as in unet_module
                                tag_prefix = f"Validation_Viz/Batch_{batch_idx}/{fname}_Slice_{slice_num}"

                                writer.add_image(
                                    f"{tag_prefix}/Output", output_viz[i], 0
                                )  # Global step 0 for single vis point
                                writer.add_image(
                                    f"{tag_prefix}/Target", target_viz[i], 0
                                )
                                writer.add_image(
                                    f"{tag_prefix}/Difference", diff_viz[i], 0
                                )

                            except Exception as e:
                                logging.warning(
                                    f"  Could not log image {i} in batch {batch_idx} for {fname} slice {slice_num}: {e}"
                                )

                        pbar.update(1)
                        batch_count += 1

            logging.info(
                f"  Finished processing {batch_count} batches for {version_name}."
            )

        except Exception as e:
            logging.error(
                f"  Error processing checkpoint {ckpt_path}: {e}", exc_info=True
            )  # Include traceback
        finally:
            if "writer" in locals() and writer:
                writer.close()
                logging.info(f"  TensorBoard writer closed for {version_name}.")

    logging.info("\nVisualization script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize model outputs from checkpoints."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./logs/unet",  # <<< Updated Default Path >>>
        help="Base directory containing experiment version folders (e.g., 'logs/unet').",
    )
    # Remove the vis_log_dir argument
    # parser.add_argument(
    #     "--vis_log_dir",
    #     type=str,
    #     default="visualization_logs",
    #     help="Directory to save the new TensorBoard visualization logs."
    # )
    parser.add_argument(
        "--data_config",
        type=str,
        default="fastmri_dirs.yaml",
        help="Path to the data configuration YAML file.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=3,
        help="Number of validation batches to visualize per checkpoint (0 for all).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for loading validation data.",
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=2,
        help="Number of rows in the image grid logged to TensorBoard.",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        default="singlecoil",
        choices=["singlecoil", "multicoil"],
        help="Challenge type.",
    )
    parser.add_argument(
        "--subsplit",
        type=str,
        default="knee",
        choices=["knee", "brain"],
        help="Dataset subsplit.",
    )

    args = parser.parse_args()
    visualize_checkpoints(args)
