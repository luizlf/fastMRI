"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch

import fastmri
from src.mri_data import CombinedSliceDataset, SliceDataset, AnnotatedSliceDataset

from typing import NamedTuple, List, Dict, Any, Tuple
import logging


# <<< Define NamedTuple for batch structure >>>
class CustomBatch(NamedTuple):
    image: torch.Tensor
    target: Optional[torch.Tensor]
    mean: torch.Tensor
    std: torch.Tensor
    fname: List[str]
    slice_num: List[int]
    max_value: torch.Tensor
    annotations: List[List[Dict[str, Any]]]  # List of lists of annotation dicts
    mask: Optional[torch.Tensor]  # Added optional mask field


# <<< Define Custom Collate Function >>>
def custom_collate_fn(batch: List[Any]) -> CustomBatch:
    """Custom collate function to handle varying annotation list lengths and potentially missing targets/masks."""
    (
        images,
        targets,
        means,
        stds,
        fnames,
        slice_nums,
        max_values,
        annotations_list,
        masks,
    ) = ([], [], [], [], [], [], [], [], [])

    has_target = False
    has_mask = False

    # Check the structure of the first element to determine access method
    if not batch:
        raise ValueError("Batch cannot be empty")

    # Assuming the dataset __getitem__ returns a tuple or dict-like object
    # Structure after transform: (image, target, mean, std, fname, slice_num, max_value, annotations, mask)
    # This needs to be consistent with the actual output of your Dataset's __getitem__ AFTER the transform

    for sample in batch:
        try:
            # --- Access data assuming a specific tuple structure ---
            # Adjust indices if your dataset/transform returns a different order or structure
            img = sample[0]
            tgt = (
                sample[1] if len(sample) > 1 else None
            )  # Handle cases where target might be missing
            mean_val = sample[2]
            std_val = sample[3]
            fname_val = sample[4]
            slice_val = sample[5]
            max_val = sample[6]
            annots = (
                sample[7] if len(sample) > 7 else []
            )  # Default to empty list if missing
            mask_val = (
                sample[8] if len(sample) > 8 else None
            )  # Handle cases where mask might be missing
            # --- End Access --- #

            # Basic validation (keep for tensors)
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Expected image tensor, got {type(img)}")
            # Remove float/int checks for mean/std/max - they are tensors
            # if not isinstance(mean_val, (float, int)): raise TypeError(f"Expected mean float/int, got {type(mean_val)}")
            # if not isinstance(std_val, (float, int)): raise TypeError(f"Expected std float/int, got {type(std_val)}")
            # if not isinstance(max_val, (float, int)): raise TypeError(f"Expected max_value float/int, got {type(max_val)}")

            # <<< Extract item() from tensors before appending >>>
            images.append(img)
            means.append(
                mean_val.item() if isinstance(mean_val, torch.Tensor) else mean_val
            )  # Get scalar
            stds.append(
                std_val.item() if isinstance(std_val, torch.Tensor) else std_val
            )  # Get scalar
            fnames.append(str(fname_val))  # Ensure string
            slice_nums.append(int(slice_val))  # Ensure int
            max_values.append(
                max_val.item() if isinstance(max_val, torch.Tensor) else max_val
            )  # Get scalar
            # Ensure annotations is a list, default to empty list if None
            annotations_list.append(annots if isinstance(annots, list) else [])

            if tgt is not None:
                if not isinstance(tgt, torch.Tensor):
                    raise TypeError(f"Expected target tensor, got {type(tgt)}")
                targets.append(tgt)
                has_target = True

            if mask_val is not None:
                if not isinstance(mask_val, torch.Tensor):
                    raise TypeError(f"Expected mask tensor, got {type(mask_val)}")
                masks.append(mask_val)
                has_mask = True

        except (
            IndexError,
            TypeError,
            AttributeError,
        ) as e:  # Added AttributeError for .item()
            logging.error(
                f"Error processing sample in custom_collate_fn: {e}. Sample data structure might be unexpected."
            )
            # Log the sample structure for debugging
            try:
                logging.error(
                    f"Problematic sample type: {type(sample)}, value: {sample}"
                )
            except Exception:
                logging.error("Could not log problematic sample details.")
            # Option: Skip sample or raise error
            # continue
            raise ValueError(
                f"Error processing sample structure in custom collate: {e}"
            )

    # Stack tensors if list is not empty
    images = torch.stack(images, 0) if images else torch.empty(0)
    means = torch.tensor(means, dtype=torch.float32) if means else torch.empty(0)
    stds = torch.tensor(stds, dtype=torch.float32) if stds else torch.empty(0)
    max_values = (
        torch.tensor(max_values, dtype=torch.float32) if max_values else torch.empty(0)
    )

    targets = torch.stack(targets, 0) if has_target and targets else None
    masks = torch.stack(masks, 0) if has_mask and masks else None

    # Return the custom batch structure
    return CustomBatch(
        image=images,
        target=targets,
        mean=means,
        std=stds,
        fname=fnames,
        slice_num=slice_nums,
        max_value=max_values,
        annotations=annotations_list,  # Keep as list of lists
        mask=masks,
    )


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[AnnotatedSliceDataset, CombinedSliceDataset] = (
        worker_info.dataset
    )  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True

    return False


class FastMriDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        only_annotated: bool = False,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            sample_rate: Fraction of slices of the training data split to use.
                Can be set to less than 1.0 for rapid prototyping. If not set,
                it defaults to 1.0. To subsample the dataset either set
                sample_rate (sample by slice) or volume_sample_rate (sample by
                volume), but not both.
            val_sample_rate: Same as sample_rate, but for val split.
            test_sample_rate: Same as sample_rate, but for test split.
            volume_sample_rate: Fraction of volumes of the training data split
                to use. Can be set to less than 1.0 for rapid prototyping. If
                not set, it defaults to 1.0. To subsample the dataset either
                set sample_rate (sample by slice) or volume_sample_rate (sample
                by volume), but not both.
            val_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            test_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            train_filter: A callable which takes as input a training example
                metadata, and returns whether it should be part of the training
                dataset.
            val_filter: Same as train_filter, but for val split.
            test_filter: Same as train_filter, but for test split.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError(
                "Can set val_sample_rate or val_volume_sample_rate, but not both."
            )
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError(
                "Can set test_sample_rate or test_volume_sample_rate, but not both."
            )

        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.only_annotated = only_annotated

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.val_filter
            elif data_partition == "test":
                sample_rate = (
                    self.test_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.test_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.test_filter

        # if desired, combine train and val together for the train split
        dataset: Union[AnnotatedSliceDataset, CombinedSliceDataset]
        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
            ]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
            )
            collate_to_use = custom_collate_fn
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / f"{self.challenge}_{data_partition}"

            # dataset = SliceDataset(
            #     root=data_path,
            #     transform=data_transform,
            #     sample_rate=sample_rate,
            #     volume_sample_rate=volume_sample_rate,
            #     challenge=self.challenge,
            #     use_dataset_cache=self.use_dataset_cache_file,
            #     raw_sample_filter=raw_sample_filter,
            # )
            dataset = AnnotatedSliceDataset(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
                subsplit="knee",
                multiple_annotation_policy="all",
                only_annotated=self.only_annotated,
            )
            collate_to_use = custom_collate_fn

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None

        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = fastmri.data.VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
            persistent_workers=True if self.num_workers >= 1 else False,
            pin_memory=False,
            collate_fn=collate_to_use,
        )
        # iter(dataloader)
        # print(next(iter(dataloader)))

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / f"{self.challenge}_test"
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                # NOTE: Fixed so that val and test use correct sample rates
                sample_rate = self.sample_rate  # if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                # _ = SliceDataset(
                #     root=data_path,
                #     transform=data_transform,
                #     sample_rate=sample_rate,
                #     volume_sample_rate=volume_sample_rate,
                #     challenge=self.challenge,
                #     use_dataset_cache=self.use_dataset_cache_file,
                # )
                _ = AnnotatedSliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                    subsplit="knee",
                    multiple_annotation_policy="all",
                    only_annotated=self.only_annotated,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform, data_partition=self.test_split
        )

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--test_split",
            choices=("val", "test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (train split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--val_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (val split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--test_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (test split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (train split only). If not "
                "given all will be used. Cannot set together with sample_rate."
            ),
        )
        parser.add_argument(
            "--val_volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (val split only). If not "
                "given all will be used. Cannot set together with val_sample_rate."
            ),
        )
        parser.add_argument(
            "--test_volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (test split only). If not "
                "given all will be used. Cannot set together with test_sample_rate."
            ),
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=bool,
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser
