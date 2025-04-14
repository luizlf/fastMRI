import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch_lightning as pl
import numpy as np

# Attempt to import PIL for image labeling
try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning(
        "PIL (Pillow) not found. Cannot add text labels to images. Install with: pip install Pillow"
    )

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.unet.unet_module import UnetModule
    from src.data_module import FastMriDataModule
    from src.mri_data import AnnotatedSliceDataset
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


def add_label_to_tensor(image_tensor, label):
    """Adds a text label to the bottom-right corner of an image tensor."""
    if not PIL_AVAILABLE:
        return image_tensor  # Return original if PIL is not available

    # Ensure tensor is on CPU, remove batch/channel dim if needed, convert to [0, 255] uint8
    # Input assumed to be [C, H, W] where C=1
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 1:
        logging.warning(
            f"Labeling expects tensor shape [1, H, W], got {image_tensor.shape}"
        )
        return image_tensor  # Return original if shape is wrong

    img_tensor_squeezed = image_tensor.squeeze(0)  # Now [H, W]

    try:
        img_np = (img_tensor_squeezed.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np, mode="L")  # L mode for grayscale
        draw = ImageDraw.Draw(pil_image)

        # Use default font or try loading a small one
        try:
            font = ImageFont.load_default()
        except IOError:
            font = ImageFont.load_default()  # Fallback just in case

        # Calculate text size and position
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:  # Older Pillow versions might not have textbbox
            text_width, text_height = draw.textsize(label, font=font)

        img_width, img_height = pil_image.size
        padding = 3  # Reduced padding
        position = (
            img_width - text_width - padding,
            img_height - text_height - padding,
        )

        # Draw text (white text)
        draw.text(position, label, fill=255, font=font)

        # Convert back to tensor [0, 1] in [C, H, W] format
        labeled_np = np.array(pil_image).astype(np.float32) / 255.0
        labeled_tensor = torch.from_numpy(labeled_np).unsqueeze(
            0
        )  # Add channel dimension back
        return labeled_tensor

    except Exception as e:
        logging.warning(f"Failed to add label '{label}' to image: {e}")
        return image_tensor  # Return original tensor [C, H, W] on error


def custom_collate_fn(batch_list):
    """Custom collate function to handle lists of annotations correctly."""
    # Separate attributes from the list of samples
    images = [item.image for item in batch_list]
    targets = [item.target for item in batch_list]
    means = [item.mean for item in batch_list]
    stds = [item.std for item in batch_list]
    fnames = [item.fname for item in batch_list]
    slice_nums = [item.slice_num for item in batch_list]
    max_values = [item.max_value for item in batch_list]
    annotations = [
        item.annotations for item in batch_list
    ]  # This will be List[List[Dict]]

    # Stack tensors
    images_tensor = torch.stack(images, 0)
    targets_tensor = torch.stack(targets, 0)
    means_tensor = torch.stack(means, 0)
    stds_tensor = torch.stack(stds, 0)
    slice_nums_tensor = torch.tensor(slice_nums, dtype=torch.int)
    max_values_tensor = torch.tensor(max_values, dtype=torch.float)

    # Return as a dictionary (or a custom class/namedtuple if preferred)
    # Using a simple class for attribute access like batch.image
    class BatchContainer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return BatchContainer(
        image=images_tensor,
        target=targets_tensor,
        mean=means_tensor,
        std=stds_tensor,
        fname=fnames,
        slice_num=slice_nums_tensor,
        max_value=max_values_tensor,
        annotations=annotations,  # Keep as List[List[Dict]]
    )


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

    # Create the *Dataset* directly, not the full DataModule here
    try:
        val_dataset = AnnotatedSliceDataset(
            root=data_path / f"{args.challenge}_val",  # Construct val path
            challenge=args.challenge,
            transform=val_transform,
            sample_rate=1.0,
            volume_sample_rate=None,
            use_dataset_cache=True,  # Use cache if available
            subsplit=args.subsplit,
            multiple_annotation_policy="all",  # Or match your training setup
            only_annotated=True,  # Match training setup if applicable
        )
        logging.info(f"Validation Dataset created with {len(val_dataset)} samples.")
    except Exception as e:
        logging.error(f"Failed to create validation dataset: {e}", exc_info=True)
        return

    # Create DataLoader using the custom collate function
    try:
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            sampler=None,  # No sampler needed for visualization
            shuffle=False,
            collate_fn=custom_collate_fn,  # <<< Use custom collate fn >>>
        )
        logging.info("Validation DataLoader created with custom collate function.")
    except Exception as e:
        logging.error(f"Failed to create DataLoader: {e}", exc_info=True)
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
                        # <<< MODIFIED TO LOG A 1x4 LABELED GRID >>>
                        actual_batch_size = output_viz.shape[0]
                        annotations_available = hasattr(batch, "annotations")
                        annotations_len = (
                            len(batch.annotations) if annotations_available else 0
                        )

                        # Check for potential mismatch or missing annotations attribute
                        if not annotations_available:
                            logging.warning(
                                f"Batch object missing 'annotations' attribute for batch {batch_idx}. Skipping ROI drawing."
                            )
                            process_annotations = False
                        elif actual_batch_size != annotations_len:
                            logging.warning(
                                f"Batch size mismatch in batch {batch_idx}: output_viz shape {actual_batch_size}, annotations length {annotations_len}. Skipping ROI drawing."
                            )
                            process_annotations = False
                        else:
                            process_annotations = True

                        for i in range(actual_batch_size):
                            try:
                                # Get individual images (already normalized [0,1], shape [1, H, W])
                                target_img_tensor = target_viz[i]
                                output_img_tensor = output_viz[i]
                                error_img_tensor = diff_viz[i]

                                # Get filename and slice number for tagging
                                fname = Path(batch.fname[i]).stem
                                slice_num = int(batch.slice_num[i].item())
                                base_tag = f"{fname}_Slice_{slice_num}"

                                # Create ROI mask (initialize) - Not needed for border
                                # roi_mask = torch.zeros_like(target_img_tensor) # Shape [1, H, W]

                                # Initialize ROI overlay tensor as a clone of target
                                roi_overlay_tensor = target_img_tensor.clone()

                                # Draw ROI border if annotations are valid
                                if process_annotations:
                                    try:
                                        annotation_list_for_sample = batch.annotations[
                                            i
                                        ]
                                        if (
                                            isinstance(annotation_list_for_sample, list)
                                            and len(annotation_list_for_sample) > 0
                                        ):
                                            annot = annotation_list_for_sample[0]
                                            if (
                                                isinstance(annot, dict)
                                                and all(
                                                    k in annot
                                                    for k in (
                                                        "x",
                                                        "y",
                                                        "width",
                                                        "height",
                                                    )
                                                )
                                                and annot.get("x", -1) != -1
                                            ):
                                                x, y, w, h = (
                                                    int(annot["x"]),
                                                    int(annot["y"]),
                                                    int(annot["width"]),
                                                    int(annot["height"]),
                                                )
                                                if w > 0 and h > 0 and PIL_AVAILABLE:
                                                    # Convert tensor [1, H, W] to uint8 PIL Image
                                                    img_tensor_squeezed = (
                                                        roi_overlay_tensor.squeeze(0)
                                                    )
                                                    img_np = (
                                                        img_tensor_squeezed.cpu().numpy()
                                                        * 255
                                                    ).astype(np.uint8)
                                                    pil_image = Image.fromarray(
                                                        img_np, mode="L"
                                                    )
                                                    draw = ImageDraw.Draw(pil_image)

                                                    # Define box coordinates, ensure they are within image bounds
                                                    x1, y1 = x, y
                                                    x2 = min(x + w, pil_image.width - 1)
                                                    y2 = min(
                                                        y + h, pil_image.height - 1
                                                    )
                                                    box = (x1, y1, x2, y2)

                                                    # Draw rectangle outline (white, 1 pixel width)
                                                    draw.rectangle(
                                                        box, outline=255, width=1
                                                    )

                                                    # Convert back to tensor [0, 1] in [1, H, W] format
                                                    labeled_np = (
                                                        np.array(pil_image).astype(
                                                            np.float32
                                                        )
                                                        / 255.0
                                                    )
                                                    roi_overlay_tensor = (
                                                        torch.from_numpy(
                                                            labeled_np
                                                        ).unsqueeze(0)
                                                    )
                                                elif not PIL_AVAILABLE:
                                                    logging.warning(
                                                        f"PIL not available, cannot draw ROI border for {base_tag}"
                                                    )

                                    # Catch specific errors if needed, or general exceptions
                                    except IndexError as idx_e:
                                        logging.warning(
                                            f"Internal IndexError accessing annotations[{i}] for {base_tag}, despite length check: {idx_e}. Skipping ROI."
                                        )
                                    except Exception as annot_e:
                                        logging.warning(
                                            f"Could not process annotations/draw ROI border for {base_tag}: {annot_e}"
                                        )

                                # Clamp just in case drawing pushed values slightly outside [0,1]
                                roi_overlay_tensor = torch.clamp(
                                    roi_overlay_tensor, 0.0, 1.0
                                )

                                # --- Add Labels --- #
                                labeled_target = add_label_to_tensor(
                                    target_img_tensor, "Target"
                                )
                                labeled_recon = add_label_to_tensor(
                                    output_img_tensor, "Reconstruction"
                                )
                                labeled_error = add_label_to_tensor(
                                    error_img_tensor, "Error"
                                )
                                labeled_roi = add_label_to_tensor(
                                    roi_overlay_tensor, "ROI Border"
                                )  # Updated label

                                # Stack labeled images for grid: Target, Recon, Error, ROI Overlay
                                images_to_grid = [
                                    labeled_target,
                                    labeled_recon,
                                    labeled_error,
                                    labeled_roi,
                                ]

                                # Create 1x4 grid (nrow=4)
                                grid = torchvision.utils.make_grid(
                                    images_to_grid, nrow=4, padding=5, pad_value=0.5
                                )

                                # Log the single grid image
                                # Use a new tag name to avoid caching issues
                                writer.add_image(
                                    f"{base_tag}/ComparisonGrid_LabeledBorder", grid, 1
                                )
                                # Log the scalar at step 1 as well (optional, can remove if not needed)
                                writer.add_scalar(f"{base_tag}/LogCheck_Border", 1, 1)

                            except Exception as e:
                                logging.warning(
                                    f"  Could not create/log grid for image {i} in batch {batch_idx} ({base_tag}): {e}"
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
