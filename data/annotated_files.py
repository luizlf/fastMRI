import os
import pandas as pd
import requests
import yaml
from pathlib import Path
from tqdm import tqdm
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Helper Functions (adapted from src/mri_data.py) ---


def download_csv(subsplit: str, annotation_version: str | None, path: Path) -> Path:
    """Downloads the annotation CSV file."""
    if path.is_file():
        logging.info(f"Annotation file already exists: {path}")
        return path

    logging.info(f"Downloading annotation file for {subsplit}...")
    path.parent.mkdir(parents=True, exist_ok=True)

    if annotation_version is None:
        url = (
            f"https://raw.githubusercontent.com/microsoft/fastmri-plus/"
            f"main/Annotations/{subsplit}.csv"
        )
    else:
        url = (
            f"https://raw.githubusercontent.com/microsoft/fastmri-plus/"
            f"{annotation_version}/Annotations/{subsplit}.csv"
        )

    try:
        response = requests.get(url, timeout=20, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading annotation file from {url}: {e}")
        raise

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc=f"Downloading {path.name}",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        disable=not total_size_in_bytes > 0,
    )

    try:
        with open(path, "wb") as fh:
            for chunk in response.iter_content(1024 * 1024):
                progress_bar.update(len(chunk))
                fh.write(chunk)
        progress_bar.close()
        logging.info(f"Successfully downloaded {path.name}")
        return path
    except Exception as e:
        logging.error(f"Error writing annotation file {path}: {e}")
        if path.exists():
            path.unlink()  # Clean up partial file
        raise


def fetch_dir(key: str, data_config_file: str | Path = "fastmri_dirs.yaml") -> Path:
    """Fetches directory path from the config file."""
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        logging.error(f"Path config file not found: {data_config_file}")
        raise FileNotFoundError(f"Path config file not found: {data_config_file}")

    try:
        with open(data_config_file, "r") as f:
            config = yaml.safe_load(f)
            data_dir = config.get(key)
            if data_dir is None:
                raise KeyError(f"Key '{key}' not found in {data_config_file}")
    except Exception as e:
        logging.error(f"Error reading config file {data_config_file}: {e}")
        raise

    path = Path(data_dir)
    if not path.is_dir():
        logging.warning(f"Directory specified by key '{key}' does not exist: {path}")
        # Optionally raise an error here if the directory must exist
        # raise NotADirectoryError(f"Directory specified by key '{key}' does not exist: {path}")
    return path


# --- Main Script Logic ---


def find_annotated_files(
    challenge: str = "singlecoil",
    subsplit: str = "knee",
    annotation_version: str | None = None,  # None uses the latest version
    config_file: str | Path = "fastmri_dirs.yaml",
):
    """Finds annotated files and calculates their total size."""

    # 1. Get annotation data
    annotation_cache_dir = Path(".annotation_cache")
    annotation_filename = f"{subsplit}{annotation_version or ''}.csv"
    annotation_path = annotation_cache_dir / annotation_filename
    try:
        annotation_path = download_csv(subsplit, annotation_version, annotation_path)
        annotations_csv = pd.read_csv(annotation_path)
    except Exception as e:
        logging.error(f"Failed to load or download annotations: {e}")
        return

    # 2. Filter for relevant annotations (slice-level with valid coordinates)
    slice_annotations = annotations_csv[
        (annotations_csv["study_level"] != "Yes")
        & (annotations_csv["x"] != -1)
        & (annotations_csv["y"] != -1)
        & (annotations_csv["width"] != -1)
        & (annotations_csv["height"] != -1)
    ]
    annotated_filenames_set = set(slice_annotations["file"].unique())
    logging.info(
        f"Found {len(annotated_filenames_set)} unique filenames with slice-level annotations."
    )

    # 3. Get dataset paths
    try:
        data_root = fetch_dir(f"{subsplit}_path", config_file)
    except (FileNotFoundError, KeyError, NotADirectoryError) as e:
        logging.error(f"Could not retrieve base data directory: {e}")
        return

    splits = ["train", "val", "test"]
    split_paths = {}
    for split in splits:
        path = data_root / f"{challenge}_{split}"
        if path.is_dir():
            split_paths[split] = path
        else:
            logging.warning(f"Directory not found for split '{split}': {path}")

    # 4. Find matching files in dataset splits and calculate size
    annotated_files_by_split = {split: [] for split in split_paths}
    total_size_bytes = 0

    logging.info("Searching for annotated files in dataset directories...")
    for split, path in split_paths.items():
        logging.info(f"Checking directory: {path}")
        found_count = 0
        try:
            for item in path.iterdir():
                if (
                    item.is_file()
                    and item.suffix == ".h5"
                    and item.stem in annotated_filenames_set
                ):
                    annotated_files_by_split[split].append(item)
                    try:
                        total_size_bytes += item.stat().st_size
                        found_count += 1
                    except OSError as e:
                        logging.warning(f"Could not get size for file {item}: {e}")
        except OSError as e:
            logging.warning(f"Could not read directory {path}: {e}")
        logging.info(f"Found {found_count} annotated files in '{split}' split.")

    # 5. Print results
    print("-" * 30)
    print("Annotated Files Found:")
    print("-" * 30)
    any_found = False
    for split, files in annotated_files_by_split.items():
        if files:
            any_found = True
            print(f"\n--- {split.upper()} Split ({len(files)} files) ---")
            for f in sorted(files):
                print(f.name)
        else:
            print(f"\n--- {split.upper()} Split (0 files) ---")

    if not any_found:
        print("\nNo annotated files found in the specified directories.")

    print("-" * 30)
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_mb / 1024
    print(f"\nTotal size of annotated files:")
    print(f"  {total_size_bytes:,} bytes")
    print(f"  {total_size_mb:.2f} MB")
    print(f"  {total_size_gb:.2f} GB")
    print("-" * 30)


def create_annotated_subset(
    target_root_dir: str | Path,
    challenge: str = "singlecoil",
    subsplit: str = "knee",
    annotation_version: str | None = None,
    config_file: str | Path = "fastmri_dirs.yaml",
):
    """Creates a new directory structure containing only annotated files
    (or all test files) copied from the original dataset.
    """
    target_root_dir = Path(target_root_dir)
    logging.info(f"Target directory for annotated subset: {target_root_dir}")

    # 1. Get annotation data (same as before)
    annotation_cache_dir = Path(".annotation_cache")
    annotation_filename = f"{subsplit}{annotation_version or ''}.csv"
    annotation_path = annotation_cache_dir / annotation_filename
    try:
        annotation_path = download_csv(subsplit, annotation_version, annotation_path)
        annotations_csv = pd.read_csv(annotation_path)
    except Exception as e:
        logging.error(f"Failed to load or download annotations: {e}")
        return

    # Filter for slice-level annotations to get the set of filenames
    slice_annotations = annotations_csv[
        (annotations_csv["study_level"] != "Yes")
        & (annotations_csv["x"] != -1)
        & (annotations_csv["y"] != -1)
        & (annotations_csv["width"] != -1)
        & (annotations_csv["height"] != -1)
    ]
    annotated_filenames_set = set(slice_annotations["file"].unique())
    logging.info(
        f"Identified {len(annotated_filenames_set)} unique filenames with annotations."
    )

    # 2. Get source dataset paths (same as before)
    try:
        source_data_root = fetch_dir(f"{subsplit}_path", config_file)
    except (FileNotFoundError, KeyError, NotADirectoryError) as e:
        logging.error(f"Could not retrieve base data directory: {e}")
        return

    splits = ["train", "val", "test"]
    source_split_paths = {}
    target_split_paths = {}

    for split in splits:
        source_path = source_data_root / f"{challenge}_{split}"
        target_path = target_root_dir / f"{challenge}_{split}"
        if source_path.is_dir():
            source_split_paths[split] = source_path
            target_split_paths[split] = target_path
            # Create target directory structure
            try:
                target_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Ensured target directory exists: {target_path}")
            except OSError as e:
                logging.error(f"Could not create target directory {target_path}: {e}")
                return  # Stop if we can't create a target dir
        else:
            logging.warning(
                f"Source directory not found for split '{split}': {source_path}"
            )

    # 3. Copy files
    logging.info("Starting file copy process...")

    for split, source_path in source_split_paths.items():
        target_path = target_split_paths[split]
        copy_count = 0
        skip_count = 0
        error_count = 0
        logging.info(f"Processing '{split}' split: {source_path} -> {target_path}")

        try:
            items_to_process = list(source_path.iterdir())  # Get list for tqdm
            with tqdm(total=len(items_to_process), desc=f"Copying {split}") as pbar:
                for source_item_path in items_to_process:
                    pbar.update(1)
                    if not source_item_path.is_file():
                        continue  # Skip directories

                    target_item_path = target_path / source_item_path.name
                    should_copy = False

                    if split == "test":
                        # Copy ALL files for the test split
                        should_copy = True
                    elif (
                        source_item_path.suffix == ".h5"
                        and source_item_path.stem in annotated_filenames_set
                    ):
                        # Copy .h5 files if they are in the annotated set for train/val
                        should_copy = True
                    else:
                        skip_count += 1

                    if should_copy:
                        try:
                            if (
                                not target_item_path.exists()
                            ):  # Avoid re-copying if run again
                                shutil.copy2(source_item_path, target_item_path)
                                copy_count += 1
                            else:
                                # logging.debug(f"Skipping existing file: {target_item_path}")
                                skip_count += 1  # Count as skipped if it exists
                        except (shutil.Error, OSError) as e:
                            logging.warning(
                                f"Could not copy {source_item_path} to {target_item_path}: {e}"
                            )
                            error_count += 1

        except OSError as e:
            logging.error(f"Could not read source directory {source_path}: {e}")
            continue  # Try next split if one fails

        logging.info(
            f"Finished '{split}' split. Copied: {copy_count}, Skipped/Exists: {skip_count}, Errors: {error_count}"
        )

    logging.info("Annotated subset creation process completed.")


if __name__ == "__main__":
    # --- Configuration ---
    CHALLENGE = "singlecoil"
    SUBSPLIT = "knee"
    ANNOTATION_VERSION = None
    CONFIG_FILE = "fastmri_dirs.yaml"
    # <<< Define the new target directory for the annotated subset >>>
    TARGET_SUBSET_DIR = "data/annotated_knee_data"
    # ---------------------

    # --- Choose Action ---
    # Set to True to run the find/list annotated files action
    RUN_FIND_ACTION = False
    # Set to True to run the create annotated subset action
    RUN_CREATE_SUBSET_ACTION = True
    # ---------------------

    if RUN_FIND_ACTION:
        print("\nRunning: Find and List Annotated Files\n" + "=" * 40)
        find_annotated_files(
            challenge=CHALLENGE,
            subsplit=SUBSPLIT,
            annotation_version=ANNOTATION_VERSION,
            config_file=CONFIG_FILE,
        )

    if RUN_CREATE_SUBSET_ACTION:
        print("\nRunning: Create Annotated Subset\n" + "=" * 40)
        create_annotated_subset(
            target_root_dir=TARGET_SUBSET_DIR,
            challenge=CHALLENGE,
            subsplit=SUBSPLIT,
            annotation_version=ANNOTATION_VERSION,
            config_file=CONFIG_FILE,
        )

    if not RUN_FIND_ACTION and not RUN_CREATE_SUBSET_ACTION:
        print(
            "No action selected. Set RUN_FIND_ACTION or RUN_CREATE_SUBSET_ACTION to True."
        )
