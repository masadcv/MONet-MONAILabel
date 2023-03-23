import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.dice import DiceLoss
from torch import Tensor
from torch.nn.common_types import _size_3_t
from torch.nn.modules.loss import _Loss
from torch.nn.modules.utils import _triple


class MultiScaleConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        scales=None,
    ) -> None:
        super(MultiScaleConv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        if scales and out_channels % len(scales) != 0:
            raise ValueError(
                "Unable to split out channels {} into {} chunks".format(
                    out_channels, len(scales)
                )
            )

        self._make_multiscale_mask(scales)

    def _make_multiscale_mask(self, scales):
        k = (
            self.kernel_size[0]
            if isinstance(self.kernel_size, tuple)
            else self.kernel_size
        )

        if scales and max(scales) <= k:
            print("MultiScaleConv3d: Using multiscales {}".format(scales))

            if min(scales) < 1:
                raise ValueError(
                    "Scales cannot be less than 1 found min_scale={}".format(
                        min(scales)
                    )
                )

            # form individual masks
            kernel_masks = []
            for s in scales:
                kcmask = torch.zeros((1, self.in_channels, k, k, k)).type(torch.float32)
                ik = (k - s) / 2
                left = math.floor(ik)
                right = k - math.ceil(ik)
                kcmask[:, :, left:right, left:right, left:right] = 1.0
                kernel_masks.append(kcmask)

            # expand masks copies into main mask tensor
            self.mask = torch.zeros_like(self.weight)
            for filt in range(self.out_channels):
                self.mask[filt, ...] = kernel_masks[
                    math.floor((filt / self.out_channels) * len(scales))
                ]
        else:
            print("MultiScaleConv3d: Not using multiscales")
            self.mask = torch.ones_like(self.weight)

    # apply any function (e.g. module.to(), module.float(), module.numpy()) to mask as well
    # help from: https://stackoverflow.com/a/57208704/798093
    def _apply(self, fn):
        super(MultiScaleConv3d, self)._apply(fn)
        self.mask = fn(self.mask)
        return self

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight * self.mask,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            input,
            weight * self.mask,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MultiScaleConv3dBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding="same",
        use_bn=False,
        activation=nn.ReLU,
        dropout=0.2,
        scales=None,
    ):
        modules = []
        modules.append(
            MultiScaleConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                scales=scales,
            )
        )
        if use_bn:
            modules.append(nn.BatchNorm3d(num_features=out_channels))
        if activation:
            modules.append(activation())
        if dropout:
            modules.append(nn.Dropout(p=dropout))

        super().__init__(*modules)


class Conv3dBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding="same",
        use_bn=False,
        activation=nn.ReLU,
        dropout=0.2,
    ):
        modules = []
        modules.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        if use_bn:
            modules.append(nn.BatchNorm3d(num_features=out_channels))
        if activation:
            modules.append(activation())
        if dropout:
            modules.append(nn.Dropout(p=dropout))

        super().__init__(*modules)


class MyDiceCELoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=ce_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same."
            )

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        if len(dice_loss.shape) < len(ce_loss.shape):
            dice_loss = dice_loss.view(*dice_loss.shape + ce_loss.shape[2:])
        total_loss: torch.Tensor = (
            self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        )

        return total_loss
