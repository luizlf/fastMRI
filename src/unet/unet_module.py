"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
import numpy as np
import logging
from pathlib import Path

import torch
from torch.nn import functional as F
from torch import nn

# Attempt to import PIL for image labeling
try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    # No logging here, will warn inside the function if needed

# from fastmri.models import Unet
from src.unet.unet import Unet

from src.mri_module import MriModule

from fastmri import evaluate  # Import evaluate functions


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


# Helper function for labeling images
def add_label_to_tensor(image_tensor, label):
    """Adds a text label to the bottom-right corner of an image tensor."""
    if not PIL_AVAILABLE:
        # Log warning only when actually called without PIL
        logging.warning(
            "PIL (Pillow) not found. Cannot add text labels to images. Install with: pip install Pillow"
        )
        return image_tensor

    # Input assumed to be [C, H, W] where C=1
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 1:
        logging.warning(
            f"Labeling expects tensor shape [1, H, W], got {image_tensor.shape}"
        )
        return image_tensor

    img_tensor_squeezed = image_tensor.squeeze(0)

    try:
        img_np = (img_tensor_squeezed.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np, mode="L")
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.load_default()
        except IOError:
            font = ImageFont.load_default()

        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(label, font=font)

        img_width, img_height = pil_image.size
        padding = 3
        position = (
            img_width - text_width - padding,
            img_height - text_height - padding,
        )

        draw.text(position, label, fill=255, font=font)

        labeled_np = np.array(pil_image).astype(np.float32) / 255.0
        labeled_tensor = torch.from_numpy(labeled_np).unsqueeze(0)
        return labeled_tensor

    except Exception as e:
        logging.warning(f"Failed to add label '{label}' to image: {e}")
        return image_tensor  # Return original tensor on error


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
        chans=64,
        num_pool_layers=3,
        drop_prob=0.0,
        lr=0.001,
        lr_factor=0.1,
        lr_patience=3,
        weight_decay=0.0,
        metric="ssim",
        roi_weight=0.5,
        attn_layer=False,
        use_roi=False,
        use_attention_gates=False,
        l1_ssim_alpha=0.85,
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
            lr_factor (float, optional): Factor by which the learning rate will
                be reduced by ReduceLROnPlateau. Defaults to 0.1.
            lr_patience (int, optional): Number of epochs with no improvement
                after which learning rate will be reduced by ReduceLROnPlateau.
                Defaults to 3.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
            roi_weight (float, optional): Weight for the region of interest (0.1 means 10% higher weight to the annotated ROI).
            l1_ssim_alpha (float, optional): Weight for L1 loss in combined L1+SSIM loss. Defaults to 0.85.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.metric = metric
        self.roi_weight = roi_weight
        self.attn_layer = attn_layer
        self.use_roi = use_roi
        self.use_attention_gates = use_attention_gates
        self.l1_ssim_alpha = l1_ssim_alpha
        # Use parent class's device handling

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
            roi_weight=self.roi_weight,
            attn_layer=self.attn_layer,
            use_attention_gates=self.use_attention_gates,
        )

        self.ssim = SSIM()

    def forward(self, image):
        return self.unet(image.unsqueeze(1)).squeeze(1)

    def create_mask(self, annotations, shape):
        """Creates a batch of binary masks from annotations.

        Args:
            annotations: The annotation data structure from the collate function.
                       Expected structure with default collate: List[List[Dict]].
            shape: The shape of the target tensor (e.g., model output) [B, H, W].

        Returns:
            A tuple containing:
              - mask (torch.Tensor): A binary mask tensor of shape [B, H, W].
              - annot_exists (bool): Whether any valid annotation existed in the batch.
        """
        B, H, W = shape
        mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        batch_annot_exists = False  # Track if any annotation exists in the batch

        # Check if annotations is a list (basic check)
        if not isinstance(annotations, list):
            logging.warning(
                f"Annotations structure unexpected (not a list), cannot create mask. Type: {type(annotations)}"
            )
            return mask, False

        # Iterate through the batch using enumerate to get batch index
        for batch_idx, annot_list in enumerate(annotations):
            # Check if batch_idx is within the mask batch dimension
            if batch_idx >= B:
                logging.warning(
                    f"Annotation list length ({len(annotations)}) exceeds mask batch size ({B}). Skipping extra annotations."
                )
                break

            # Expect annot_list to be a list containing one dictionary
            if isinstance(annot_list, list) and len(annot_list) > 0:
                annot = annot_list[0]
                if isinstance(annot, dict):
                    # Check for valid annotation keys and coordinates
                    if (
                        all(k in annot for k in ("x", "y", "width", "height"))
                        and annot.get("x", -1) != -1
                    ):
                        x, y, w, h = (
                            int(annot["x"]),
                            int(annot["y"]),
                            int(annot["width"]),
                            int(annot["height"]),
                        )

                        # Check for valid dimensions
                        if w > 0 and h > 0:
                            # Use exact annotation coordinates, clamped to image dimensions
                            y1 = max(0, y)
                            y2 = min(y + h, H)  # Use dynamic shape H
                            x1 = max(0, x)
                            x2 = min(x + w, W)  # Use dynamic shape W

                            if y2 > y1 and x2 > x1:  # Ensure valid slice after clamping
                                mask[batch_idx, y1:y2, x1:x2] = True
                                batch_annot_exists = (
                                    True  # Mark that at least one valid annot was found
                                )
                        # else: logging.debug(f"Annotation with non-positive w/h skipped: {annot}")
                    # else: logging.debug(f"Placeholder or invalid annotation dict skipped: {annot}")
                # else: logging.debug(f"Item inside annotation list is not a dict: {annot}")
            # else: logging.debug(f"Annotation list for sample {batch_idx} is not a list or is empty: {annot_list}")

        return (
            mask.to(self.device),
            batch_annot_exists,
        )  # Ensure mask is on correct device

    def training_step(self, batch, batch_idx):
        output = self(batch.image)
        target = batch.target

        mask, annot_exists = self.create_mask(batch.annotations, output.shape)
        mask = mask.to(output.device)
        log_to_logger = self.trainer is not None and self.trainer.logger is not None

        # --- Calculate L1 Components --- #
        loss_image_l1 = F.l1_loss(output, target)
        loss_mask_roi_l1 = torch.tensor(0.0, device=output.device)
        final_l1_component = loss_image_l1

        if self.use_roi and annot_exists and mask.sum() > 0:
            mask_float = mask.float()
            loss_mask_roi_l1 = F.l1_loss(output[mask], target[mask])
            final_l1_component = (loss_image_l1 * 0.5) + (loss_mask_roi_l1 * 0.5)

        # --- Calculate SSIM Component (always on full image) --- #
        output_ssim = output.unsqueeze(1)
        target_ssim = target.unsqueeze(1)
        max_value_ssim = batch.max_value.view(-1)
        loss_image_ssim = 1.0 - self.ssim(output_ssim, target_ssim, max_value_ssim)

        # --- Combine Losses --- #
        loss = (self.l1_ssim_alpha * final_l1_component) + (
            (1 - self.l1_ssim_alpha) * loss_image_ssim
        )

        # --- Logging --- #
        self.log(
            "train_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "train_loss_l1",
            final_l1_component.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "train_loss_ssim",
            loss_image_ssim.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "train_loss_image_l1",
            loss_image_l1.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "train_loss_roi_l1",
            loss_mask_roi_l1.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "train_annot_exists",
            float(annot_exists),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "train_mask_sum",
            mask.sum().float(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        target = batch.target
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        mask, annot_exists = self.create_mask(batch.annotations, output.shape)
        mask = mask.to(output.device)
        has_roi_pixels = annot_exists and mask.sum() > 0
        log_to_logger = self.trainer is not None and self.trainer.logger is not None

        # --- Always Calculate L1 Components --- #
        val_loss_image_l1 = F.l1_loss(output, target)
        val_loss_roi_l1 = (
            F.l1_loss(output[mask], target[mask])
            if has_roi_pixels
            else torch.tensor(0.0, device=output.device)
        )
        # --- Calculate L1 Loss --- #
        val_loss_l1 = val_loss_image_l1 * (1 - self.roi_weight) + val_loss_roi_l1 * (
            self.roi_weight
        )

        # --- Always Calculate SSIM Components --- #
        output_ssim = output.unsqueeze(1)
        target_ssim = target.unsqueeze(1)
        max_value_ssim = batch.max_value.view(-1)
        val_loss_image_ssim = 1.0 - self.ssim(output_ssim, target_ssim, max_value_ssim)
        val_loss_roi_ssim = (
            1.0
            - self.ssim(
                output_ssim,
                target_ssim,
                max_value_ssim,
                mask=mask.unsqueeze(1).float(),
                use_roi=True,
            )
            if has_roi_pixels
            else torch.tensor(0.0, device=output.device)
        )
        # --- Calculate SSIM Loss --- #
        val_loss_ssim = val_loss_image_ssim * (
            1 - self.roi_weight
        ) + val_loss_roi_ssim * (self.roi_weight)

        # --- Always Calculate Combined L1+SSIM Components --- #
        val_loss_image_l1_ssim = (self.l1_ssim_alpha * val_loss_image_l1) + (
            (1 - self.l1_ssim_alpha) * val_loss_image_ssim
        )
        val_loss_roi_l1_ssim = (
            (self.l1_ssim_alpha * val_loss_roi_l1)
            + ((1 - self.l1_ssim_alpha) * val_loss_roi_ssim)
            if has_roi_pixels
            else torch.tensor(0.0, device=output.device)
        )

        # --- Calculate Combined L1+SSIM Loss --- #
        val_loss_l1_ssim = val_loss_image_l1_ssim * (
            1 - self.roi_weight
        ) + val_loss_roi_l1_ssim * (self.roi_weight)

        # --- Determine Primary val_loss based on Experiment Config --- #
        if self.metric == "l1":
            if self.use_roi and has_roi_pixels:
                # Weighted average for L1 ROI focus
                val_loss = val_loss_l1
            else:
                # Use full image L1 if not using ROI or no ROI exists
                val_loss = val_loss_image_l1
        elif self.metric == "l1_ssim":
            if self.use_roi and has_roi_pixels:
                # Weighted average for L1+SSIM ROI focus
                val_loss = val_loss_l1_ssim
            else:
                # Use full image L1+SSIM if not using ROI or no ROI exists
                val_loss = val_loss_image_l1_ssim
        else:
            # Fallback or error for unknown metric
            logging.warning(
                f"Unknown metric '{self.metric}' in validation_step, using image L1 loss."
            )
            val_loss = val_loss_image_l1

        # --- Logging (Log ALL calculated metrics) --- #
        # Log the primary monitored loss
        self.log(
            "val_loss",
            val_loss.detach(),
            on_epoch=True,
            prog_bar=True,  # Keep this on prog bar
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        # Log individual and combined components
        self.log(
            "val_loss_image_l1",
            val_loss_image_l1.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "val_loss_roi_l1",
            val_loss_roi_l1.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "val_loss_l1",
            val_loss_l1.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "val_loss_image_ssim",
            val_loss_image_ssim.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "val_loss_roi_ssim",
            val_loss_roi_ssim.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "val_loss_ssim",
            val_loss_ssim.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "val_loss_image_l1_ssim",
            val_loss_image_l1_ssim.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "val_loss_roi_l1_ssim",
            val_loss_roi_l1_ssim.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )
        self.log(
            "val_loss_l1_ssim",
            val_loss_l1_ssim.detach(),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=log_to_logger,
            batch_size=batch.image.size(0),
        )

        # --- Image/Metric Logging (Keep previous logic for PSNR/SSIM/NMSE via MriModule) ---
        # <<< Re-add visualization block >>>
        num_batches_to_log = 1
        if (
            batch_idx < num_batches_to_log
            and self.logger is not None
            and hasattr(self.logger, "experiment")
        ):
            try:
                import torchvision

                # Detach, move to CPU, denormalize
                # Use the already calculated output/target if available, otherwise recalculate
                # Note: output/target here are normalized, viz needs denormalization or normalization to [0,1]
                output_viz = (output * std + mean).cpu().detach()
                target_viz = (target * std + mean).cpu().detach()

                # Add channel dimension if missing (B, H, W) -> (B, 1, H, W)
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

                # Calculate difference
                diff_viz = torch.abs(output_viz - target_viz)

                # --- Log a 1x5 labeled grid per image in the batch ---
                actual_batch_size = output_viz.shape[0]
                annotations_available = hasattr(batch, "annotations")
                annotations_len = len(batch.annotations) if annotations_available else 0

                # Check for annotation consistency
                process_annotations = annotations_available and (
                    actual_batch_size == annotations_len
                )
                if not process_annotations and annotations_available:
                    logging.debug(
                        f"Batch size mismatch for viz: images {actual_batch_size}, annotations {annotations_len}. Skipping ROI drawing."
                    )
                elif not annotations_available:
                    logging.debug(
                        "Batch missing annotations attribute. Skipping ROI drawing."
                    )

                for i in range(actual_batch_size):
                    try:
                        # Get individual images [1, H, W]
                        target_img_tensor = target_viz[i]
                        output_img_tensor = output_viz[i]
                        error_img_tensor = diff_viz[i]

                        fname_str = batch.fname[i]
                        slice_num_int = batch.slice_num[i]
                        base_tag = f"{Path(fname_str).stem}_Slice_{slice_num_int}"

                        # --- Get the generated mask for this sample --- #
                        mask_slice_tensor = mask[i].float().unsqueeze(0)
                        if mask_slice_tensor.ndim == 2:
                            mask_slice_tensor = mask_slice_tensor.unsqueeze(0)
                        elif (
                            mask_slice_tensor.ndim == 3
                            and mask_slice_tensor.shape[0] != 1
                        ):
                            mask_slice_tensor = mask_slice_tensor[0].unsqueeze(0)

                        # --- Create ROI Border Image --- #
                        roi_overlay_tensor = target_img_tensor.clone()
                        if process_annotations:
                            try:
                                # ... (rest of ROI border drawing logic using PIL) ...
                                annotation_list_for_sample = batch.annotations[i]
                                if (
                                    isinstance(annotation_list_for_sample, list)
                                    and len(annotation_list_for_sample) > 0
                                ):
                                    annot = annotation_list_for_sample[0]
                                    if (
                                        isinstance(annot, dict)
                                        and all(
                                            k in annot
                                            for k in ("x", "y", "width", "height")
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
                                            img_np_roi = (
                                                roi_overlay_tensor.squeeze(0)
                                                .cpu()
                                                .numpy()
                                                * 255
                                            ).astype(np.uint8)
                                            pil_image_roi = Image.fromarray(
                                                img_np_roi, mode="L"
                                            )
                                            draw_roi = ImageDraw.Draw(pil_image_roi)
                                            x1, y1 = x, y
                                            x2, y2 = min(
                                                x + w, pil_image_roi.width - 1
                                            ), min(y + h, pil_image_roi.height - 1)
                                            draw_roi.rectangle(
                                                (x1, y1, x2, y2), outline=255, width=1
                                            )
                                            labeled_np_roi = (
                                                np.array(pil_image_roi).astype(
                                                    np.float32
                                                )
                                                / 255.0
                                            )
                                            roi_overlay_tensor = torch.from_numpy(
                                                labeled_np_roi
                                            ).unsqueeze(0)
                                        elif not PIL_AVAILABLE:
                                            logging.warning(
                                                f"PIL not available, cannot draw ROI border for {base_tag}"
                                            )
                            except Exception as annot_e:
                                logging.warning(
                                    f"Could not process annotations/draw ROI border for {base_tag} in training validation: {annot_e}"
                                )
                        roi_overlay_tensor = torch.clamp(roi_overlay_tensor, 0.0, 1.0)

                        # --- Add Labels & Create Grid (Update to 5 images) --- #
                        labeled_target = add_label_to_tensor(
                            target_img_tensor, "Target"
                        )
                        labeled_recon = add_label_to_tensor(
                            output_img_tensor, "Reconstruction"
                        )
                        labeled_error = add_label_to_tensor(error_img_tensor, "Error")
                        labeled_roi = add_label_to_tensor(
                            roi_overlay_tensor, "ROI Border"
                        )

                        images_to_grid = [
                            labeled_target,
                            labeled_recon,
                            labeled_error,
                            labeled_roi,
                        ]
                        grid = torchvision.utils.make_grid(
                            images_to_grid, nrow=4, padding=5, pad_value=0.5
                        )

                        # --- Log the Grid --- #
                        self.logger.experiment.add_image(
                            f"{base_tag}/ComparisonGrid", grid, self.current_epoch
                        )

                    except Exception as e:
                        logging.warning(
                            f"Could not create/log validation grid for image {i} in batch {batch_idx}, epoch {self.current_epoch}: {e}"
                        )

            except ImportError:
                logging.warning(
                    "torchvision not found, cannot log image grids. Skipping image logging."
                )
            except Exception as e:
                logging.warning(
                    f"Could not log validation images for batch {batch_idx} in epoch {self.current_epoch}: {e}"
                )
        # <<< End of re-added visualization block >>>

        # Return values for aggregation (include all calculated losses)
        # MriModule uses output/target/max_value for PSNR/SSIM aggregation.
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": val_loss,  # Primary monitored loss
            "val_loss_image_l1": val_loss_image_l1,
            "val_loss_roi_l1": val_loss_roi_l1,
            "val_loss_l1": val_loss_l1,
            "val_loss_image_ssim": val_loss_image_ssim,
            "val_loss_roi_ssim": val_loss_roi_ssim,
            "val_loss_ssim": val_loss_ssim,
            "val_loss_image_l1_ssim": val_loss_image_l1_ssim,
            "val_loss_roi_l1_ssim": val_loss_roi_l1_ssim,
            "val_loss_l1_ssim": val_loss_l1_ssim,
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

        mask, annot_exist = self.create_mask(batch.annotations, output.shape)
        factor = mask.numel() / mask.sum()
        if not annot_exist:
            factor = 1
        val_loss = 1 - SSIM()(
            output, batch.target, batch.max_value, mask=mask, use_roi=True
        )
        image_l1_loss = F.l1_loss(output, batch.target)
        image_ssim_loss = 1 - SSIM()(output, batch.target, batch.max_value)
        if annot_exist:
            roi_l1_loss = F.l1_loss(output * mask, batch.target * mask) * factor
            roi_ssim_loss = 1 - SSIM()(
                output, batch.target, batch.max_value, mask=mask, use_roi=True
            )
        else:
            roi_l1_loss = torch.tensor(0)
            roi_ssim_loss = torch.tensor(0)

        # print(f"val_step_comp Validation Loss: {val_loss}, image loss: {image_l1_loss}, roi loss: {roi_l1_loss}")
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": val_loss,
            "image_l1_loss": image_l1_loss,
            "image_ssim_loss": image_ssim_loss,
            "roi_l1_loss": roi_l1_loss / factor,
            "roi_ssim_loss": roi_ssim_loss,
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        # Create output dictionary
        # <<< Direct access to fname and slice_num >>>
        # Assuming test_step_outputs expects dicts per sample
        # We need to iterate through the batch here
        actual_batch_size = output.shape[0]
        for i in range(actual_batch_size):
            step_output = {
                "fname": batch.fname[i],
                "slice": batch.slice_num[i],
                "output": (output[i] * std[i] + mean[i])
                .cpu()
                .numpy(),  # Index output/std/mean
            }

            # Append to the instance list (inherited from MriModule)
            if hasattr(self, "test_step_outputs"):
                self.test_step_outputs.append(step_output)
            else:
                # Fallback or warning if the attribute wasn't initialized (should not happen with correct MriModule init)
                print(
                    "Warning: self.test_step_outputs not found in UnetModule instance."
                )

        # test_step typically doesn't return anything, results are collected in self.test_step_outputs
        # return # No return needed

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

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
            "--lr_factor",
            default=0.1,
            type=float,
            help="Factor for ReduceLROnPlateau scheduler",
        )
        parser.add_argument(
            "--lr_patience",
            default=3,
            type=int,
            help="Patience for ReduceLROnPlateau scheduler",
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
        parser.add_argument(
            "--l1_ssim_alpha",
            default=0.85,
            type=float,
            help="Weight for L1 loss in combined L1+SSIM loss (default: 0.85)",
        )

        return parser
