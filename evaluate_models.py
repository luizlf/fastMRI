# Create evaluate_models.py

import argparse
import logging
import sys
import yaml
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.patches as patches

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Attempt to import project modules and dependencies
try:
    from src.unet.unet_module import UnetModule
    from src.mri_data import AnnotatedSliceDataset
    from src import transforms as T
    from fastmri import evaluate
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError as e:
    print(f"Error importing project modules or dependencies: {e}")
    print(
        "Ensure fastmri, PyTorch, pandas, numpy, matplotlib, seaborn, Pillow are installed."
    )
    print(f"Project root added to path: {project_root}")
    PIL_AVAILABLE = False  # Set flag if PIL failed
    # Allow script to continue if other imports worked, PIL check is done later
    if "PIL" not in str(e):
        sys.exit(1)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Helper Functions ---


def fetch_dir(key: str, data_config_file: str | Path) -> Path:
    """Fetches directory path from the config file."""
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        raise FileNotFoundError(f"Path config file not found: {data_config_file}")
    with open(data_config_file, "r") as f:
        config = yaml.safe_load(f)
        data_dir = config.get(key)
        if data_dir is None:
            raise KeyError(f"Key '{key}' not found in {data_config_file}")
    path = Path(data_dir)
    # Check specifically for the directory needed later
    # if not path.is_dir():
    #     logging.warning(f"Directory specified by key '{key}' does not exist: {path}")
    return path


def custom_collate_fn(batch_list):
    """Custom collate function to handle lists of annotations correctly."""
    # (Assume this function is defined exactly as in visualize_checkpoints.py)
    # Separate attributes from the list of samples
    items = {
        "image": [],
        "target": [],
        "mean": [],
        "std": [],
        "fname": [],
        "slice_num": [],
        "max_value": [],
        "annotations": [],
    }
    skipped_attrs = set()

    for item in batch_list:
        for key in items:
            if hasattr(item, key):
                items[key].append(getattr(item, key))
            elif key not in skipped_attrs:
                # Log only once per missing attribute type
                logging.debug(
                    f"Sample missing attribute '{key}', returning None/empty for batch."
                )
                skipped_attrs.add(key)

    # Handle potential missing attributes gracefully
    images_tensor = torch.stack(items["image"], 0) if items["image"] else torch.empty(0)
    targets_tensor = (
        torch.stack(items["target"], 0) if items["target"] else torch.empty(0)
    )
    means_tensor = torch.stack(items["mean"], 0) if items["mean"] else torch.empty(0)
    stds_tensor = torch.stack(items["std"], 0) if items["std"] else torch.empty(0)
    slice_nums_tensor = (
        torch.tensor(items["slice_num"], dtype=torch.int)
        if items["slice_num"]
        else torch.empty(0, dtype=torch.int)
    )
    max_values_tensor = (
        torch.tensor(items["max_value"], dtype=torch.float)
        if items["max_value"]
        else torch.empty(0, dtype=torch.float)
    )

    class BatchContainer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return BatchContainer(
        image=images_tensor,
        target=targets_tensor,
        mean=means_tensor,
        std=stds_tensor,
        fname=items["fname"],  # List
        slice_num=slice_nums_tensor,
        max_value=max_values_tensor,
        annotations=items["annotations"],  # List[List[Dict]]
    )


def get_roi_mask(annotations_list_for_sample, shape_hw):
    """Creates a binary ROI mask tensor from potentially multiple annotations."""
    mask = torch.zeros(shape_hw, dtype=torch.bool)  # H, W
    # annotations_list_for_sample is List[Dict]
    if isinstance(annotations_list_for_sample, list):
        for (
            annot
        ) in (
            annotations_list_for_sample
        ):  # Iterate through all annotations for the slice
            if (
                isinstance(annot, dict)
                and all(k in annot for k in ("x", "y", "width", "height"))
                and annot.get("x", -1) != -1
            ):
                x, y, w, h = (
                    int(annot["x"]),
                    int(annot["y"]),
                    int(annot["width"]),
                    int(annot["height"]),
                )
                if w > 0 and h > 0:
                    y_end = min(y + h, shape_hw[0])
                    x_end = min(x + w, shape_hw[1])
                    mask[y:y_end, x:x_end] = True  # Use OR logic for overlapping ROIs
    return mask


def calculate_metrics(target, output, max_value, roi_mask=None):
    """Calculates PSNR, SSIM, NMSE, optionally masked by ROI."""
    target_np = target.numpy()
    output_np = output.numpy()

    # <<< Recalculate maxval from target data for robust PSNR >>>
    max_val_for_psnr = target_np.max()
    # Ensure max_val is positive for PSNR calculation
    if max_val_for_psnr <= 0:
        logging.debug(
            f"Target max value is non-positive ({max_val_for_psnr:.4f}), PSNR will be -inf or NaN."
        )
        # PSNR is undefined or -inf if maxval is 0 or target/output are constant zero.
        # Assign NaN or handle as per desired reporting.
        psnr_val = np.nan
        psnr_roi_val = np.nan
    else:
        psnr_val = evaluate.psnr(target_np, output_np, maxval=max_val_for_psnr)
        # We still use the original max_value from HDF5 for SSIM as it's often expected by that metric.
        # Assign NaN or handle as per desired reporting.
        ssim_val = evaluate.ssim(
            target_np[None, ...], output_np[None, ...], maxval=max_value
        )
        nmse_val = evaluate.nmse(target_np, output_np)

    metrics = {"psnr": psnr_val, "ssim": ssim_val, "nmse": nmse_val}

    if roi_mask is not None and roi_mask.sum() > 0:
        roi_mask_np = roi_mask.numpy()
        # Also use recalculated maxval for ROI PSNR
        if max_val_for_psnr > 0:
            # Calculate on masked data points
            target_roi_np = target_np[roi_mask_np]
            output_roi_np = output_np[roi_mask_np]
            # Recalculate max specifically for the ROI region might be even more accurate
            # max_val_roi = target_roi_np.max()
            # if max_val_roi <= 0: psnr_roi_val = np.nan
            # else: psnr_roi_val = evaluate.psnr(target_roi_np, output_roi_np, maxval=max_val_roi)
            # --- OR --- use overall target max for consistency:
            psnr_roi_val = evaluate.psnr(
                target_roi_np, output_roi_np, maxval=max_val_for_psnr
            )

            metrics["psnr_roi"] = psnr_roi_val
            metrics["nmse_roi"] = evaluate.nmse(target_roi_np, output_roi_np)
        else:
            metrics["psnr_roi"] = np.nan
            metrics["nmse_roi"] = evaluate.nmse(
                target_np[roi_mask_np], output_np[roi_mask_np]
            )  # NMSE doesn't need maxval
    else:
        metrics["psnr_roi"] = np.nan
        metrics["nmse_roi"] = np.nan

    return metrics


# --- Main Evaluation Function ---


def evaluate_models(args):
    """Loads models, evaluates metrics, generates visualizations."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    logging.info(f"Evaluation results will be saved to: {output_dir}")

    # --- Setup Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    logging.info(f"Loading {args.eval_split} data...")
    try:
        data_root = fetch_dir(f"{args.subsplit}_path", args.data_config)
        eval_data_path = data_root / f"{args.challenge}_{args.eval_split}"
        if not eval_data_path.is_dir():
            raise FileNotFoundError(
                f"Evaluation split directory not found: {eval_data_path}"
            )

        # Create transform (no mask needed for evaluation usually)
        eval_transform = T.UnetDataTransform(
            which_challenge=args.challenge, mask_func=None
        )

        eval_dataset = AnnotatedSliceDataset(
            root=eval_data_path,
            challenge=args.challenge,
            transform=eval_transform,
            sample_rate=1.0,
            use_dataset_cache=True,
            subsplit=args.subsplit,
            multiple_annotation_policy="all",
            only_annotated=True,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        logging.info(
            f"Loaded {args.eval_split} dataset with {len(eval_dataset)} samples."
        )
    except Exception as e:
        logging.error(f"Failed to load data: {e}", exc_info=True)
        return

    # --- Load Baseline Model ---
    baseline_ckpt_path = Path(args.baseline_checkpoint)
    if not baseline_ckpt_path.is_file():
        logging.error(f"Baseline checkpoint not found: {baseline_ckpt_path}")
        return
    try:
        logging.info(f"Loading baseline model from: {baseline_ckpt_path}")
        baseline_model = UnetModule.load_from_checkpoint(
            checkpoint_path=str(baseline_ckpt_path), map_location=device
        )
        baseline_model.eval()
        baseline_model.to(device)
        baseline_name = (
            baseline_ckpt_path.parent.parent.name
        )  # e.g., version_0 or unet_baseline_l1
        logging.info(f"Baseline model '{baseline_name}' loaded.")
    except Exception as e:
        logging.error(f"Failed to load baseline model: {e}", exc_info=True)
        return

    # --- Load Comparison Models ---
    comparison_models = {}
    for ckpt_path_str in args.comparison_checkpoints:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.is_file():
            logging.warning(f"Comparison checkpoint not found, skipping: {ckpt_path}")
            continue
        try:
            logging.info(f"Loading comparison model from: {ckpt_path}")
            model = UnetModule.load_from_checkpoint(
                checkpoint_path=str(ckpt_path), map_location=device
            )
            model.eval()
            model.to(device)
            model_name = ckpt_path.parent.parent.name  # Get version/experiment name
            if model_name in comparison_models:  # Avoid name collision
                model_name = f"{model_name}_{ckpt_path.stem}"
            comparison_models[model_name] = model
            logging.info(f"Comparison model '{model_name}' loaded.")
        except Exception as e:
            logging.warning(
                f"Failed to load comparison model {ckpt_path}, skipping: {e}",
                exc_info=True,
            )

    if not comparison_models:
        logging.error("No valid comparison models loaded. Exiting.")
        return

    # --- Evaluation Loop ---
    results = []
    num_batches_to_process = (
        args.num_batches if args.num_batches > 0 else len(eval_dataloader)
    )
    processed_samples = 0

    logging.info(f"Starting evaluation loop for {num_batches_to_process} batches...")
    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(eval_dataloader),
            total=num_batches_to_process,
            desc="Evaluating Batches",
        ):
            if batch_idx >= num_batches_to_process:
                break
            if (
                batch.image.numel() == 0
            ):  # Skip empty batches if collate fn returned empty
                logging.warning(f"Skipping empty batch {batch_idx}")
                continue

            # Move data
            image_dev = batch.image.to(device)
            target_dev = batch.target.to(device)
            # Remove batch mean/std - will recalculate per slice
            # mean_dev = batch.mean.to(device)
            # std_dev = batch.std.to(device)
            max_value_dev = batch.max_value.to(device)

            # --- Inference ---
            baseline_output_dev = baseline_model(image_dev)
            comparison_outputs_dev = {
                name: model(image_dev) for name, model in comparison_models.items()
            }

            # --- Process Results per Slice ---
            actual_batch_size = image_dev.shape[0]
            for i in range(actual_batch_size):
                # Extract slice data and move to CPU
                target_slice = target_dev[i].squeeze().cpu()
                baseline_output_slice = baseline_output_dev[i].squeeze().cpu()
                # Remove batch mean/std extraction
                # mean_slice = mean_dev[i].item()
                # std_slice = std_dev[i].item()
                max_value_slice = max_value_dev[i].item()
                fname = batch.fname[i]
                slice_num = batch.slice_num[i].item()
                annotations_list = (
                    batch.annotations[i]
                    if hasattr(batch, "annotations") and i < len(batch.annotations)
                    else None
                )

                # <<< Recalculate mean and std directly from target slice >>>
                mean_slice = target_slice.mean().item()
                std_slice = target_slice.std().item()
                # Add epsilon to std to prevent division by zero
                epsilon = 1e-6
                std_slice = std_slice if std_slice > epsilon else epsilon

                # Denormalize using calculated mean/std
                target_slice_denorm = target_slice * std_slice + mean_slice
                baseline_output_slice_denorm = (
                    baseline_output_slice * std_slice + mean_slice
                )

                # Convert to numpy for plotting
                target_np = target_slice_denorm.numpy()
                baseline_np = baseline_output_slice_denorm.numpy()

                # Get ROI mask (handles multiple annots) and List of ROI coordinates for plotting
                roi_mask_slice = get_roi_mask(annotations_list, target_slice.shape)
                roi_coords_list = []  # <<< Store multiple coords >>>
                if isinstance(annotations_list, list):
                    for annot in annotations_list:
                        if (
                            isinstance(annot, dict)
                            and all(k in annot for k in ("x", "y", "width", "height"))
                            and annot.get("x", -1) != -1
                        ):
                            if int(annot["width"]) > 0 and int(annot["height"]) > 0:
                                roi_coords_list.append(
                                    (
                                        int(annot["x"]),
                                        int(annot["y"]),
                                        int(annot["width"]),
                                        int(annot["height"]),
                                    )
                                )

                # --- Calculate Baseline Metrics (uses combined mask) ---
                baseline_metrics = calculate_metrics(
                    target_slice_denorm,
                    baseline_output_slice_denorm,
                    max_value_slice,
                    roi_mask_slice,  # Combined mask
                )
                results.append(
                    {
                        "filename": fname,
                        "slice": slice_num,
                        "model": baseline_name,
                        **baseline_metrics,
                    }
                )

                # --- Process Each Comparison Model ---
                for (
                    model_name,
                    model_output_dev_batch,
                ) in comparison_outputs_dev.items():
                    output_slice = model_output_dev_batch[i].squeeze().cpu()
                    # <<< Log normalized mean >>>
                    logging.debug(
                        f"Slice {slice_num}, Model '{model_name}': output_slice mean (normalized)={output_slice.mean():.4f}"
                    )

                    output_slice_denorm = output_slice * std_slice + mean_slice
                    model_np = output_slice_denorm.numpy()

                    # Calculate Metrics (uses combined mask)
                    comp_metrics = calculate_metrics(
                        target_slice_denorm,
                        output_slice_denorm,
                        max_value_slice,
                        roi_mask_slice,  # Combined mask
                    )
                    results.append(
                        {
                            "filename": fname,
                            "slice": slice_num,
                            "model": model_name,
                            **comp_metrics,
                        }
                    )

                    # <<< Debug Print (uses denormalized mean) >>>
                    logging.debug(
                        f"Generating viz for: slice={slice_num}, model_name='{model_name}', model_np_mean={model_np.mean():.4f}"
                    )

                    # --- Generate Visualizations --- #
                    if processed_samples < args.max_vis_samples:
                        try:
                            # Error maps
                            baseline_error_map = torch.abs(
                                baseline_output_slice_denorm - target_slice_denorm
                            )
                            model_error_map = torch.abs(
                                output_slice_denorm - target_slice_denorm
                            )
                            error_diff_map = baseline_error_map - model_error_map

                            # Determine color limits for heatmap (symmetric around 0)
                            max_abs_diff = torch.max(torch.abs(error_diff_map)).item()
                            if max_abs_diff == 0:
                                max_abs_diff = 1e-6
                            vmin, vmax = -max_abs_diff, max_abs_diff

                            # Determine display range based on Target Min/Max
                            plot_vmin = target_np.min()
                            plot_vmax = target_np.max()
                            if plot_vmax <= plot_vmin:
                                plot_vmax = plot_vmin + 1e-6

                            # --- Create Plots using Matplotlib --- #
                            plt.style.use("dark_background")
                            fig_c, axes_c = plt.subplots(1, 4, figsize=(20, 5))
                            plot_data = [
                                target_np,
                                baseline_np,
                                model_np,
                                error_diff_map.numpy(),
                            ]
                            plot_titles = [
                                "Target",
                                f"Baseline ({baseline_name})",
                                f"Model ({model_name})",
                                "Error Difference",
                            ]
                            cmaps = ["gray", "gray", "gray", "coolwarm"]

                            for k, (ax, data, title, cmap) in enumerate(
                                zip(axes_c.flat, plot_data, plot_titles, cmaps)
                            ):
                                if title == "Error Difference":
                                    im = ax.imshow(
                                        data, cmap=cmap, vmin=vmin, vmax=vmax
                                    )
                                else:
                                    im = ax.imshow(
                                        data, cmap=cmap, vmin=plot_vmin, vmax=plot_vmax
                                    )

                                ax.set_title(title)
                                ax.axis("off")

                                # <<< Add ALL ROI borders using Matplotlib patches >>>
                                if roi_coords_list:  # Check if the list is not empty
                                    for (
                                        roi_coords
                                    ) in roi_coords_list:  # Iterate through coords
                                        x, y, w, h = roi_coords
                                        rect = patches.Rectangle(
                                            (x, y),
                                            w,
                                            h,
                                            linewidth=1.5,
                                            edgecolor="lime",
                                            facecolor="none",
                                        )
                                        ax.add_patch(rect)

                            fig_c.suptitle(
                                f"Comparison: {Path(fname).stem}_s{slice_num}",
                                fontsize=16,
                                color="white",
                            )
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                            # Save comparison grid using model_name for uniqueness
                            grid_filename = (
                                vis_dir
                                / f"{Path(fname).stem}_s{slice_num}_{model_name}_vs_{baseline_name}_comparison.png"
                            )
                            fig_c.savefig(
                                grid_filename, bbox_inches="tight", facecolor="black"
                            )
                            plt.close(fig_c)

                        except Exception as vis_e:
                            logging.warning(
                                f"Failed to generate visualization for {fname} slice {slice_num}: {vis_e}",
                                exc_info=True,
                            )

                processed_samples += 1

    # --- Aggregate and Save Metrics ---
    if not results:
        logging.warning("No results were generated.")
        return

    results_df = pd.DataFrame(results)
    metrics_filename = output_dir / "evaluation_metrics.csv"
    results_df.to_csv(metrics_filename, index=False)
    logging.info(f"Saved detailed metrics to {metrics_filename}")

    # Calculate and print average metrics with highlighting
    try:
        avg_metrics = results_df.groupby("model")[
            ["psnr", "ssim", "nmse", "psnr_roi", "nmse_roi"]
        ].mean()

        # --- Highlight Best Values --- #
        higher_is_better = ["psnr", "ssim", "psnr_roi"]
        lower_is_better = ["nmse", "nmse_roi"]
        best_indices = {}

        for metric in avg_metrics.columns:
            if metric in higher_is_better:
                # Use skipna=True to handle potential NaNs in ROI metrics
                best_indices[metric] = avg_metrics[metric].idxmax(skipna=True)
            elif metric in lower_is_better:
                best_indices[metric] = avg_metrics[metric].idxmin(skipna=True)

        # ANSI codes for bold text
        BOLD = "\033[1m"
        RESET = "\033[0m"

        # Format the output manually for better alignment
        print("\n--- Average Metrics (Best in Bold) ---")
        col_width = 12  # Define column width
        header = f"{'model':<30}"  # Adjust model name spacing if needed
        for col in avg_metrics.columns:
            header += f" {col:>{col_width}}"
        print(header)
        print("-" * len(header))

        # Create rows with manual padding
        for model_name, row_data in avg_metrics.iterrows():
            row_str = f"{model_name:<30}"
            for metric, value in row_data.items():
                scalar_value = value
                if isinstance(value, (np.ndarray, list)) and len(value) == 1:
                    scalar_value = value[0]

                if pd.isna(scalar_value):
                    value_str = "NaN"
                else:
                    try:
                        value_str = f"{scalar_value:.4f}"
                    except (TypeError, ValueError):
                        value_str = str(scalar_value)

                is_best = best_indices.get(metric) == model_name

                # Calculate padding needed
                padding = col_width - len(value_str)
                padded_value_str = " " * padding + value_str

                if is_best and pd.notna(scalar_value):
                    # Add bold codes *around* the padded string
                    row_str += f" {BOLD}{padded_value_str}{RESET}"
                else:
                    # Add the padded string
                    row_str += f" {padded_value_str}"
            print(row_str)

        print("-" * len(header))
        # Adjust length of final separator if needed
        print("-" * len(header) + "\n")

    except Exception as e:
        logging.error(
            f"Could not calculate or print average metrics: {e}", exc_info=True
        )
        # Fallback to simple print if formatting fails
        if "avg_metrics" in locals():
            print("\n--- Average Metrics (Raw) ---")
            print(avg_metrics.to_string(float_format=lambda x: f"{x:.4f}"))
            print("---------------------------\n")

    logging.info("Evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare U-Net models against a baseline."
    )
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        required=True,
        help="Path to the baseline model .ckpt file.",
    )
    parser.add_argument(
        "--comparison_checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to one or more comparison model .ckpt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save metrics and visualizations.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="fastmri_dirs.yaml",
        help="Path to the data configuration YAML file.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Max number of batches to process (0 for all).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_vis_samples",
        type=int,
        default=50,
        help="Maximum number of slices to generate visualizations for.",
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
    evaluate_models(args)
