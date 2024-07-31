"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from torch import nn

# from fastmri.models import Unet
from src.unet.unet import Unet

from src.mri_module import MriModule


class SSIM(nn.Module):
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range, mask=None, use_roi=False):
        data_range = data_range[:, None, None, None]

        if use_roi:
            X = X * mask
            Y = Y * mask

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        w = self.w.to(X.device)
        ux = F.conv2d(X, w)
        uy = F.conv2d(Y, w)
        uxx = F.conv2d(X * X, w)
        uyy = F.conv2d(Y * Y, w)
        uxy = F.conv2d(X * Y, w)

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        """
        if mask is not None:
            mask_resized = F.interpolate(
                mask.unsqueeze(1).float(),
                size=torch.Size([314, 314]),
                mode="bilinear",
                align_corners=False,
            )
            mask_resized = mask_resized.squeeze(1)
            S = S * mask_resized
        """

        return S.mean()


class UnetModule(MriModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        metric="ssim",
        roi_weight=0.1,
        attn_layer=False,
        use_roi=False,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
            roi_weight (float, optional): Weight for the region of interest (0.1 means 10% higher weight to the annotated ROI).
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.metric = metric
        self.roi_weight = roi_weight
        self.attn_layer = attn_layer
        self.use_roi = use_roi

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
            roi_weight=self.roi_weight,
            attn_layer=self.attn_layer,
        )

        if self.metric == "ssim":
            self.ssim = SSIM()

    def forward(self, image):
        return self.unet(image.unsqueeze(1)).squeeze(1)

    def create_mask(self, annotations, shape, device):
        annot_exists = True
        mask = torch.zeros(shape, device=device)
        for annotation in annotations:
            if annotation["x"].item() == -1:
                pass
                # mask = torch.ones(shape, device=device)
            else:
                # mask = torch.ones(shape, device=device)
                x, y, w, h = (
                    annotation["x"],
                    annotation["y"],
                    annotation["width"],
                    annotation["height"],
                )
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    # mask[..., y : y + h, x : x + w] += self.roi_weight
                    mask[..., y : y + h, x : x + w] = 1
        if torch.all(mask == 0):
            annot_exists = False
        #     mask = torch.ones(shape, device=device)
        return mask, annot_exists

    def training_step(self, batch, batch_idx):
        output = self(batch.image)
        if self.metric == "l1":
            if self.use_roi:
                mask, annot_exists = self.create_mask(batch.annotations, output.shape, output.device)
                if annot_exists:
                    factor = mask.numel() / mask.sum()
                else:
                    factor = 1
                loss_mask = F.l1_loss(output * mask, batch.target * mask) * factor
                loss_image = F.l1_loss(output, batch.target) 
                loss = loss_image + loss_mask
                # print('batch.target * mask max: ', (batch.target * mask).max())
                # print('batch.target * mask min: ', (batch.target * mask).min())
            else:
                loss = F.l1_loss(output, batch.target) 
        elif self.metric == "ssim":
            mask, _ = self.create_mask(batch.annotations, output.shape, output.device)
            loss = 1 - self.ssim(output, batch.target, batch.max_value, mask=mask, use_roi=self.use_roi)

        self.log("loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        if self.metric == "l1":
            # val_loss = F.l1_loss(output, batch.target)
            if self.use_roi:
                mask, annot_exists = self.create_mask(batch.annotations, output.shape, output.device)
                if annot_exists:
                    factor = mask.numel() / mask.sum()
                else:
                    factor = 1
                val_loss_mask = F.l1_loss(output * mask, batch.target * mask) * factor
                val_loss_image = F.l1_loss(output, batch.target) 
                val_loss = val_loss_image + val_loss_mask
            else:
                val_loss = F.l1_loss(output, batch.target)
        elif self.metric == "ssim":
            mask, _ = self.create_mask(batch.annotations, output.shape, output.device)
            val_loss = 1 - self.ssim(output, batch.target, batch.max_value, mask=mask, use_roi=self.use_roi)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            # "val_loss": F.l1_loss(output, batch.target),
            "val_loss": val_loss,
        }
    

    def validation_step_comparison(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        # if self.metric == "l1":
        #     mask, annot_exists = self.create_mask(batch.annotations, output.shape, output.device)
        #     if annot_exists:
        #         factor = mask.numel() / mask.sum()
        #     else:
        #         factor = 1
        #     val_loss_mask = F.l1_loss(output * mask, batch.target * mask) * factor
        #     val_loss_image = F.l1_loss(output, batch.target) 
        #     val_loss = val_loss_image + val_loss_mask

        # elif self.metric == "ssim":
        #     mask, _ = self.create_mask(batch.annotations, output.shape, output.device)
        #     val_loss = 1 - self.ssim(output, batch.target, batch.max_value, mask=mask, use_roi=True)

        mask, _ = self.create_mask(batch.annotations, output.shape, output.device)
        val_loss = 1 - SSIM()(output, batch.target, batch.max_value, mask=mask, use_roi=True)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": val_loss,
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--roi_weight",
            default=0.1,
            type=float,
            help="Weight for the region of interest (0.1 means 10 percent higher weight to the annotated ROI)",
        )
        parser.add_argument(
            "--attn_layer",
            default=False,
            type=bool,
            help="Add attention layer to the U-Net",
        )
        parser.add_argument(
            "--use_roi",
            default=False,
            type=bool,
            help="Use region of interest mask",
        )

        return parser
