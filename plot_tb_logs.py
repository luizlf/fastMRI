import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import re
import logging

# Note: Caching requires the 'pyarrow' library. Install with: pip install pyarrow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Matplotlib / Seaborn Configuration ---
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],  # Common sans-serif fonts
        "axes.grid": True,  # Enable gridlines
        "grid.linestyle": "--",  # Dashed gridlines
        "grid.alpha": 0.6,  # Lighter gridlines
        "grid.color": "#cccccc",  # Light gray color
        "figure.facecolor": "white",  # Set figure background to white
        "axes.facecolor": "white",  # Set axes background to white
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "figure.dpi": 100,
    }
)
sns.set_palette("tab10")  # Use a standard color palette

# --- Configuration ---
LOG_DIR_ROOT = Path("logs/unet")
OUTPUT_DIR = Path("analysis")
CACHE_FILE = OUTPUT_DIR / "tb_data_cache.parquet"
# Map base loss name to the IMAGE validation metric to plot
IMAGE_LOSS_TYPES = {
    "L1 Loss": "val_loss_image_l1",
    "L1+SSIM Loss": "val_loss_image_l1_ssim",
}
# Map base loss name to the ROI validation metric to plot
ROI_LOSS_TYPES = {
    "L1 Loss": "val_loss_roi_l1",
    "L1+SSIM Loss": "val_loss_roi_l1_ssim",
}
# Define ALL potentially relevant validation metrics for caching
ALL_RELEVANT_VAL_METRICS = [
    "val_loss_image_l1",
    "val_loss_roi_l1",
    "val_loss_image_l1_ssim",
    "val_loss_roi_l1_ssim",
]
SMOOTHING_WINDOW = 7

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_experiment_name(dir_name):
    """Parses the directory name to extract model configuration details."""
    parts = dir_name.split("_")
    model_type = "unknown"
    use_roi = False
    loss_type = "unknown"
    attn_type = "none"

    # Identify base model type
    if "baseline" in parts:
        model_type = "Baseline"
    elif "cbam" in parts:
        model_type = "CBAM"
    elif "attention" in parts and "gates" in parts:
        model_type = "Attention Gates"
        if "full" in parts:
            model_type = "Full Attention"  # Overrides if 'full' is also present
    elif "full" in parts and "attention" in parts:
        model_type = "Full Attention"  # In case order varies

    # Check for ROI
    if "roi" in parts:
        use_roi = True

    # Check for attention mechanisms explicitly
    if "attn" in parts:
        attn_type = "CBAM"
        if "agate" in parts:
            attn_type = "Full"  # Both present
    elif "agate" in parts:
        attn_type = "AG"  # Only Attention Gates

    # Infer loss type
    if "l1" in parts and "ssim" in parts:
        loss_type = "L1+SSIM"
    elif "l1" in parts:
        loss_type = "L1"
    elif "ssim" in parts:  # Less likely standalone, but check
        loss_type = "SSIM"

    # Refine model name based on attention and ROI for clarity
    full_legend_name = model_type
    base_legend_name = model_type  # Start base name
    if model_type != "Baseline":
        if attn_type == "CBAM":
            full_legend_name += " (CBAM)"
            base_legend_name += " (CBAM)"
        elif attn_type == "AG":
            full_legend_name += " (AG)"
            base_legend_name += " (AG)"
        elif attn_type == "Full":
            full_legend_name += " (Full Attn)"
            base_legend_name += " (Full Attn)"

    if use_roi:
        full_legend_name += " + ROI Loss"
    else:
        full_legend_name += " (Image Loss Only)"

    # Use the refined full name for identification if needed, but base name for hue
    legend_name = full_legend_name

    return {
        "dir_name": dir_name,
        "model_type": model_type,
        "use_roi": use_roi,
        "loss_type": loss_type,
        "attn_type": attn_type,
        "legend_name": legend_name,  # Keep the detailed name if needed elsewhere
        "base_legend_name": base_legend_name,  # Use this for plot hue
    }


def extract_scalar_data(event_file_path, scalar_tags):
    """Extracts scalar data for specified tags from a TensorBoard event file."""
    data = {tag: [] for tag in scalar_tags}
    try:
        ea = event_accumulator.EventAccumulator(
            str(event_file_path), size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        available_tags = ea.Tags()["scalars"]

        for tag in scalar_tags:
            if tag in available_tags:
                events = ea.Scalars(tag)
                # Store wall_time, step (epoch), and value
                data[tag] = [(e.wall_time, e.step, e.value) for e in events]
            else:
                logging.warning(f"Tag '{tag}' not found in {event_file_path}")
    except Exception as e:
        logging.error(f"Error processing {event_file_path}: {e}", exc_info=True)
    return data


# --- Main Processing ---

# Check if cache file exists
if CACHE_FILE.is_file():
    logging.info(f"Loading data from cache file: {CACHE_FILE}")
    try:
        df = pd.read_parquet(CACHE_FILE)
        logging.info(f"Successfully loaded {len(df)} records from cache.")
        data_loaded_from_cache = True
    except Exception as e:
        logging.warning(
            f"Could not load cache file {CACHE_FILE}: {e}. Re-extracting data."
        )
        data_loaded_from_cache = False
else:
    logging.info("Cache file not found. Extracting data from TensorBoard logs.")
    data_loaded_from_cache = False

if not data_loaded_from_cache:
    # Update data_dict initialization to include base_legend_name
    data_dict = {
        "dir_name": [],
        "model_type": [],
        "use_roi": [],
        "loss_type": [],
        "attn_type": [],
        "legend_name": [],
        "base_legend_name": [],  # Added base name
        "metric": [],
        "step": [],
        "value": [],
        "wall_time": [],
    }

    # Use the comprehensive list for extraction
    scalar_tags_to_extract = ALL_RELEVANT_VAL_METRICS

    logging.info(f"Scanning log directory: {LOG_DIR_ROOT}")
    logging.info(f"Extracting tags: {scalar_tags_to_extract}")

    # Iterate through experiment directories
    for exp_dir in LOG_DIR_ROOT.iterdir():
        if exp_dir.is_dir():
            logging.info(f"Processing experiment: {exp_dir.name}")
            config = parse_experiment_name(exp_dir.name)

            event_files = list(exp_dir.glob("events.out.tfevents.*"))
            if not event_files:
                logging.warning(f"No event file found in {exp_dir}, skipping.")
                continue

            # Process ALL event files
            for event_file_path in event_files:
                logging.info(f"  Reading event file: {event_file_path.name}")
                try:
                    scalar_data = extract_scalar_data(
                        event_file_path, scalar_tags_to_extract
                    )
                    for tag, events in scalar_data.items():
                        num_events = len(events)
                        if num_events > 0:
                            data_dict["dir_name"].extend(
                                [config["dir_name"]] * num_events
                            )
                            data_dict["model_type"].extend(
                                [config["model_type"]] * num_events
                            )
                            data_dict["use_roi"].extend(
                                [config["use_roi"]] * num_events
                            )
                            data_dict["loss_type"].extend(
                                [config["loss_type"]] * num_events
                            )
                            data_dict["attn_type"].extend(
                                [config["attn_type"]] * num_events
                            )
                            data_dict["legend_name"].extend(
                                [config["legend_name"]] * num_events
                            )
                            data_dict["base_legend_name"].extend(
                                [config["base_legend_name"]] * num_events
                            )
                            cleaned_tag = str(tag).strip()
                            data_dict["metric"].extend([cleaned_tag] * num_events)
                            wall_times, steps, values = zip(*events)
                            data_dict["step"].extend(steps)
                            data_dict["value"].extend(values)
                            data_dict["wall_time"].extend(wall_times)
                except Exception as e:
                    logging.error(
                        f"    Error processing event file {event_file_path}: {e}"
                    )

    if not data_dict["value"]:  # Check if any values were actually added
        logging.error("No data extracted. Check log directory and metric names.")
        exit()

    # Convert the dictionary of lists to DataFrame
    df = pd.DataFrame(data_dict)
    logging.info(f"Extracted {len(df)} data points from logs.")

    # Adjust steps AFTER EXTRACTION, BEFORE CACHING
    mask_ag_roi = (df["base_legend_name"] == "Attention Gates (AG)") & (
        df["use_roi"] == True
    )
    if mask_ag_roi.any():
        original_max_step = df.loc[mask_ag_roi, "step"].max()
        df.loc[mask_ag_roi, "step"] = df.loc[mask_ag_roi, "step"] * 2
        new_max_step = df.loc[mask_ag_roi, "step"].max()
        logging.info(
            f"Adjusted 'step' x2 for Attention Gates (AG) + ROI. Original max: {original_max_step}, New max: {new_max_step}"
        )

    # Save to cache file (Now includes ALL extracted metrics and adjusted steps)
    try:
        logging.info(f"Saving comprehensive data to cache file: {CACHE_FILE}")
        df.to_parquet(CACHE_FILE, index=False)
        logging.info("Successfully saved cache file.")
    except Exception as e:
        logging.warning(f"Could not save cache file {CACHE_FILE}: {e}")

# Data Cleaning and Preparation (Run whether loaded from cache or extracted)

df = df.sort_values(by=["base_legend_name", "metric", "step"])  # Sort by step

# Apply smoothing
if SMOOTHING_WINDOW > 1:
    # Group by base_legend_name for smoothing
    df["value_smoothed"] = df.groupby(["base_legend_name", "metric"])[
        "value"
    ].transform(lambda s: s.rolling(SMOOTHING_WINDOW, min_periods=1).mean())
    value_col = "value_smoothed"
    logging.info(f"Applied smoothing with window size {SMOOTHING_WINDOW}")
else:
    value_col = "value"

logging.info(f"Metrics available for plotting: {df['metric'].unique()}")


# --- Plotting Loop 1: Image Loss Comparison --- #
# Define desired legend order (moved here to be accessible to both loops)
LEGEND_ORDER = [
    "Baseline",
    "CBAM (CBAM)",
    "Attention Gates (AG)",
    "Full Attention (Full Attn)",
]

logging.info("Generating Image Loss Comparison Plots...")
for base_loss_name, image_metric in IMAGE_LOSS_TYPES.items():
    # Create subplots, share Y axis if comparing the same metric
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # --- Left Plot: Trained WITHOUT ROI --- #
    ax_left = axes[0]
    plot_df_no_roi = df[
        (df["metric"] == image_metric) & (df["use_roi"] == False)
    ].copy()
    if not plot_df_no_roi.empty:
        sns.lineplot(
            data=plot_df_no_roi,
            x="step",
            y=value_col,
            hue="base_legend_name",  # Use base name for color/legend ID
            ax=ax_left,
            legend="full",  # Keep legend on this axis
            linewidth=1.5,
        )
        # Get handles/labels
        handles, labels = ax_left.get_legend_handles_labels()
        # Hide the automatically generated legend
        ax_left.get_legend().remove()

        # Reorder handles and labels based on LEGEND_ORDER
        label_handle_map = dict(zip(labels, handles))
        ordered_handles = []
        ordered_labels = []
        for label in LEGEND_ORDER:
            if label in label_handle_map:
                ordered_handles.append(label_handle_map[label])
                ordered_labels.append(label)
            # else: logging.warning(...) # Optional warning if label not found

        # Position the ORDERED legend inside the plot with desired font sizes
        if ordered_handles:
            ax_left.legend(
                ordered_handles,
                ordered_labels,
                title="Base Model Configuration",
                loc="upper right",
                fontsize=12,
                title_fontsize=13,
            )  # Increased item font size
        else:
            logging.warning(
                f"Could not create ordered legend for {base_loss_name} left plot."
            )

        ax_left.set_title(
            f"Validation {base_loss_name} (Trained without ROI)", fontsize=16
        )
        ax_left.set_xlabel("Training Step", fontsize=14)
        ax_left.set_ylabel(f"{base_loss_name} (Image Only)", fontsize=14)
        # Increase tick label size
        ax_left.tick_params(axis="both", which="major", labelsize=12)
    else:
        ax_left.set_title(
            f"Validation {base_loss_name} (Trained without ROI) - No Data", fontsize=16
        )
        ax_left.tick_params(axis="both", which="major", labelsize=12)
        ax_left.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax_left.transAxes,
        )

    # --- Right Plot: Trained WITH ROI --- #
    ax_right = axes[1]
    plot_df_roi = df[(df["metric"] == image_metric) & (df["use_roi"] == True)].copy()
    if not plot_df_roi.empty:
        sns.lineplot(
            data=plot_df_roi,
            x="step",
            y=value_col,
            hue="base_legend_name",  # Use base name for color/legend ID
            ax=ax_right,
            legend=False,  # No legend needed here
            linewidth=1.5,
        )
        # Restore title and labels with larger font
        ax_right.set_title(
            f"Validation {base_loss_name} (Trained with ROI)", fontsize=16
        )
        ax_right.set_xlabel("Training Step", fontsize=14)
        ax_right.set_ylabel("")  # Shared Y axis
        # Increase tick label size
        ax_right.tick_params(axis="both", which="major", labelsize=12)
    else:
        # Restore title and labels even if no data
        ax_right.set_title(
            f"Validation {base_loss_name} (Trained with ROI) - No Data", fontsize=16
        )
        ax_right.set_xlabel("Training Step", fontsize=14)
        ax_right.set_ylabel("")  # Shared Y axis
        ax_right.tick_params(axis="both", which="major", labelsize=12)
        ax_right.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax_right.transAxes,
        )

    # No main title
    plt.tight_layout(rect=[0, 0, 1, 1.0])

    # Save plot
    plot_filename = (
        OUTPUT_DIR
        / f"validation_{base_loss_name.lower().replace('+', '').replace(' ', '_')}_image_loss_train_roi_comparison.png"
    )
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    logging.info(f"Saved plot: {plot_filename}")
    plt.close(fig)

# --- Plotting Loop 2: ROI Loss Comparison --- #
logging.info("Generating ROI Loss Comparison Plots...")
for base_loss_name, roi_metric in ROI_LOSS_TYPES.items():
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    handles, labels = [], []

    # --- Left Plot: Trained WITHOUT ROI --- #
    ax_left = axes[0]
    # Filter for ROI metric, but models trained WITHOUT ROI loss term
    plot_df_no_roi = df[(df["metric"] == roi_metric) & (df["use_roi"] == False)].copy()
    if not plot_df_no_roi.empty:
        sns.lineplot(
            data=plot_df_no_roi,
            x="step",
            y=value_col,
            hue="base_legend_name",
            ax=ax_left,
            legend="full",
            linewidth=1.5,
        )
        # Capture and order legend from this plot
        handles, labels = ax_left.get_legend_handles_labels()
        ax_left.get_legend().remove()
        label_handle_map = dict(zip(labels, handles))
        ordered_handles = []
        ordered_labels = []
        for label in LEGEND_ORDER:
            if label in label_handle_map:
                ordered_handles.append(label_handle_map[label])
                ordered_labels.append(label)
        if ordered_handles:
            ax_left.legend(
                ordered_handles,
                ordered_labels,
                title="Base Model Configuration",
                loc="upper right",
                fontsize=12,
                title_fontsize=13,
            )
        # Update title and labels for ROI loss plot
        ax_left.set_title(
            f"Validation {base_loss_name} (ROI) (Trained without ROI)", fontsize=16
        )
        ax_left.set_xlabel("Training Step", fontsize=14)
        ax_left.set_ylabel(f"{base_loss_name} (ROI Weighted)", fontsize=14)
        ax_left.tick_params(axis="both", which="major", labelsize=12)
    else:
        ax_left.set_title(
            f"Validation {base_loss_name} (ROI) (Trained without ROI) - No Data",
            fontsize=16,
        )
        ax_left.tick_params(axis="both", which="major", labelsize=12)
        ax_left.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax_left.transAxes,
        )

    # --- Right Plot: Trained WITH ROI --- #
    ax_right = axes[1]
    # Filter for ROI metric, and models trained WITH ROI loss term
    plot_df_roi = df[(df["metric"] == roi_metric) & (df["use_roi"] == True)].copy()
    if not plot_df_roi.empty:
        sns.lineplot(
            data=plot_df_roi,
            x="step",
            y=value_col,
            hue="base_legend_name",
            ax=ax_right,
            legend=False,
            linewidth=1.5,
        )
        ax_right.set_title(
            f"Validation {base_loss_name} (ROI) (Trained with ROI)", fontsize=16
        )
        ax_right.set_xlabel("Training Step", fontsize=14)
        ax_right.set_ylabel("")  # Shared Y axis
        ax_right.tick_params(axis="both", which="major", labelsize=12)
    else:
        ax_right.set_title(
            f"Validation {base_loss_name} (ROI) (Trained with ROI) - No Data",
            fontsize=16,
        )
        ax_right.set_xlabel("Training Step", fontsize=14)
        ax_right.set_ylabel("")
        ax_right.tick_params(axis="both", which="major", labelsize=12)
        ax_right.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax_right.transAxes,
        )

    plt.tight_layout(rect=[0, 0, 1, 1.0])
    # Update filename for ROI loss plot
    plot_filename = (
        OUTPUT_DIR
        / f"validation_{base_loss_name.lower().replace('+', '').replace(' ', '_')}_roi_loss_train_roi_comparison.png"
    )
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    logging.info(f"Saved plot: {plot_filename}")
    plt.close(fig)

logging.info("Plotting complete.")
