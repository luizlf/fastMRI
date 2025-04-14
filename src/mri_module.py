"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from torch._C import device
from torchmetrics.metric import Metric

import fastmri
from fastmri import evaluate


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "quantity",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        self._device_type = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def validation_step_end(self, val_logs):
        # check inputs
        # print('\n ---------------------------------------')
        # print('VALIDATION STEP END')
        # print('\n ---------------------------------------')
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            np.random.seed(42)

            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                ####
                fname = val_logs["fname"][i]
                slice_num = int(val_logs["slice_num"][i])
                key = f"{fname}_slice_{slice_num}"
                ####
                # key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output),
                dtype=torch.float32,
                device=self._device_type,
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target)),
                dtype=torch.float32,
                device=self._device_type,
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval),
                dtype=torch.float32,
                device=self._device_type,
            ).view(1)
            max_vals[fname] = maxval

        val_logs = {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
        }
        self.validation_step_outputs.append(val_logs)
        return val_logs

        # return {
        #     "val_loss": val_logs["val_loss"],
        #     "mse_vals": dict(mse_vals),
        #     "target_norms": dict(target_norms),
        #     "ssim_vals": dict(ssim_vals),
        #     "max_vals": max_vals,
        # }

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def on_validation_epoch_end(self):
        # aggregate losses - No longer needed for val_loss, handled by self.log in validation_step
        # losses = [] # Removed
        # No longer need to manually collect these for logging averages
        # image_losses = []
        # roi_losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        val_logs = self.validation_step_outputs  # [-1]

        # print('\nval_logs: ', val_logs)
        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            # print('\nval_log type: ', type(val_log))
            # losses.append(val_log["val_loss"].view(-1))

            # ['batch_idx', 'fname', 'slice_num', 'max_value', 'output', 'target', 'val_loss']
            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

            # No longer need to manually collect these
            # if "val_loss_image" in val_log:
            #     image_losses.append(val_log["val_loss_image"].view(-1))  # type: ignore
            # if "val_loss_roi" in val_log:
            #     roi_losses.append(val_log["val_loss_roi"].view(-1))  # type: ignore

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]),
                dtype=torch.float32,
                device=self._device_type,
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()]),
                dtype=torch.float32,
                device=self._device_type,
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=self._device_type
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()]),
                dtype=torch.float32,
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        # Log epoch-level val_loss automatically computed by Lightning
        # No need for manual aggregation here for val_loss
        # if losses:
        #     val_loss = self.ValLoss(torch.sum(torch.cat(losses), dtype=torch.float32, device=self._device_type))
        #     tot_slice_examples = self.TotSliceExamples(
        #     torch.tensor(len(losses), dtype=torch.float32, device=self._device_type)
        # )
        #     self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        # else:
        #     val_loss = self.ValLoss(torch.tensor(0.0, dtype=torch.float32, device=self._device_type))
        #     tot_slice_examples = self.TotSliceExamples(torch.tensor(0.0, dtype=torch.float32, device=self._device_type))
        #     self.log("validation_loss", torch.tensor(0.0, dtype=torch.float32, device=self._device_type), prog_bar=True)
        # raise ValueError(f"{losses} {metrics} {val_logs}")
        # else:
        # Set defaults when no validation loss is available
        #    val_loss = self.ValLoss(torch.tensor(0.0, dtype=torch.float32, device=self._device_type))
        #    tot_slice_examples = self.TotSliceExamples(torch.tensor(0.0, dtype=torch.float32, device=self._device_type))
        #   self.log("validation_loss", torch.tensor(0.0, dtype=torch.float32, device=self._device_type), prog_bar=True)
        # if image_losses:
        #     overall_loss = torch.sum(
        #         torch.cat(image_losses), dtype=torch.float32, device=self._device_type
        #     )
        #     self.log(
        #         "val_metrics/overall_l1",
        #         overall_loss / tot_slice_examples,
        #         prog_bar=True,
        #     )
        # if roi_losses:
        #     overall_loss = torch.sum(
        #         torch.cat(roi_losses), dtype=torch.float32, device=self._device_type
        #     )
        #     self.log(
        #         "val_metrics/overall_l1_roi",
        #         overall_loss / tot_slice_examples,
        #         prog_bar=True,
        #     )
        # for metric, value in metrics.items():
        #     self.log(f"val_metrics/{metric}", value / tot_examples)

        if (
            tot_examples > 0
        ):  # Avoid division by zero if validation didn't run or no examples
            for metric, value in metrics.items():
                self.log(f"val_metrics/{metric}", value / tot_examples)

        self.validation_step_outputs.clear()

    def on_test_epoch_start(self):
        self.test_step_outputs.clear()

    def on_test_epoch_end(self):
        outputs = defaultdict(dict)

        for log in self.test_step_outputs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructions to {save_path}")

        fastmri.save_reconstructions(outputs, save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=16,
            type=int,
            help="Number of images to log to Tensorboard",
        )

        return parser
