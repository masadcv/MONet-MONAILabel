import copy
import logging
import math
import os
from typing import List

import numpy as np
import torch
import tqdm
from monai.config import KeysCollection
from monai.losses import DiceLoss
from monai.transforms import MapTransform
from monailabel.scribbles.transforms import InteractiveSegmentationTransform
from skimage.util.shape import view_as_windows
from torchgaussianfilter import GaussianFilter2d, GaussianFilter3d

from lib.layers import MyDiceCELoss
from lib.utils import (
    create_drop_prob_mask,
    create_drop_value_mask,
    get_eps,
    make_egd_adaptive_weights,
    make_uncertainity_from_logits,
    maxflow,
)

from .online_model import (
    OnlineModelMLPFCNHaarFeatures,
    OnlineModelMLPFCNLearnedFeatures,
    OnlineModelMLPMultiScaleFCNLearnedFeatures,
)

logger = logging.getLogger(__name__)


class MyInteractiveSegmentationTransform(InteractiveSegmentationTransform):
    def __init__(self, meta_key_postfix):
        super().__init__(meta_key_postfix)

    def _fetch_data(self, data, key):
        if key not in data.keys():
            raise ValueError(
                "Key {} not found, present keys {}".format(key, data.keys())
            )

        return data[key]

    def _get_spacing(self, d, key):
        spacing = None
        src_key = "_".join([key, self.meta_key_postfix])
        if src_key in d.keys() and "affine" in d[src_key]:
            spacing = (np.sqrt(np.sum(np.square(d[src_key]["affine"]), 0))[:-1]).astype(
                np.float32
            )

        return spacing

    def _normalise_range(self, data, b_min=0.0, b_max=1.0):
        if b_min > b_max:
            raise ValueError("b_min cannot be greater than b_max")

        a_min, a_max = data.min(), data.max()

        if (a_max - a_min) == 0.0:
            raise ValueError("Unable to normalise as min=max")

        # normalise in range [0, 1]
        norm_data = (data - a_min) / (a_max - a_min)

        # renormalise in range [b_min, b_max]
        norm_data = norm_data * (b_max - b_min) + b_min

        return norm_data

    def _copy_affine(self, d, src, dst):
        # make keys
        src_key = "_".join([src, self.meta_key_postfix])
        dst_key = "_".join([dst, self.meta_key_postfix])

        # check if keys exists, if so then copy affine info
        if src_key in d.keys() and "affine" in d[src_key]:
            # create a new destination dictionary if needed
            d[dst_key] = {} if dst_key not in d.keys() else d[dst_key]

            # copy over affine information
            d[dst_key]["affine"] = copy.deepcopy(d[src_key]["affine"])

        if src_key in d.keys() and "pixdim" in d[src_key]:
            # create a new destination dictionary if needed
            d[dst_key] = {} if dst_key not in d.keys() else d[dst_key]

            # copy over affine information
            d[dst_key]["pixdim"] = copy.deepcopy(d[src_key]["pixdim"])

        return d


class MakeLikelihoodFromScribblesECONetd(MyInteractiveSegmentationTransform):
    """
    Make Likelihood using Efficient Convolutional Online Likelihood Network (ECONet)
    from paper:

    Asad, Muhammad, Lucas Fidon, and Tom Vercauteren.
    "ECONet: Efficient convolutional online likelihood network for scribble-based interactive segmentation."
    International Conference on Medical Imaging with Deep Learning. PMLR, 2022. (link: https://arxiv.org/pdf/2201.04584.pdf)

    """

    def __init__(
        self,
        image: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "prob",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        weigh_scribbles: bool = False,
        model: str = "FEAT",
        loss: str = "CE",
        epochs: int = 200,
        lr: float = 0.01,
        lr_step: float = [0.1],
        lr_scheduler: str = "MULTI",
        dropout: float = 0.3,
        hidden_layers: List[int] = [32, 16],
        kernel_size: int = 9,
        num_filters: int = 128,
        num_layers: int = 1,
        train_feat: bool = True,
        use_argmax: bool = False,
        model_path: str = None,
        loss_threshold: float = 0.0,  # 0.005,
        use_amp: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__(meta_key_postfix)
        self.image = image
        self.scribbles = scribbles
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.weigh_scribbles = weigh_scribbles
        self.post_proc_label = post_proc_label
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.lr_step = lr_step
        self.lr_scheduler = lr_scheduler
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.train_feat = train_feat
        self.use_argmax = use_argmax
        self.model_path = model_path
        self.loss_threshold = loss_threshold
        self.use_amp = use_amp
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # at the moment only works for binary seg problem
        num_classes = 2

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.post_proc_label)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        scribbles = self._fetch_data(d, self.scribbles)
        if self.weigh_scribbles and self.scribbles + "_orig" in d.keys():
            scribbles_mask = np.squeeze(self._fetch_data(d, self.scribbles + "_orig"))
            scribbles_mask = (scribbles_mask == self.scribbles_bg_label) | (
                scribbles_mask == self.scribbles_fg_label
            )
            scribbles_mask = scribbles_mask.astype(np.float32)
        else:
            if self.weigh_scribbles:
                logging.info(
                    "WARNING: No original scribbles volume found, setting self.weigh_scribbles=False"
                )
                self.weigh_scribbles = False

        image = np.squeeze(image)
        scribbles = np.squeeze(scribbles)

        # zero-pad input image volume
        pad_size = int(math.floor(self.kernel_size / 2))
        image = np.pad(image, ((pad_size, pad_size),) * 3, mode="symmetric")

        # extract patches and select only relevant patches with scribble labels for online training
        image_patches = view_as_windows(
            image,
            (self.kernel_size, self.kernel_size, self.kernel_size),
            step=1,
        )

        # select relevant patches only for training network
        fg_patches = image_patches[scribbles == self.scribbles_fg_label]
        bg_patches = image_patches[scribbles == self.scribbles_bg_label]

        all_sel_patches = np.expand_dims(
            np.concatenate([fg_patches, bg_patches], axis=0), axis=1
        )
        all_sel_labels = np.concatenate(
            [
                np.ones((fg_patches.shape[0], 1, 1, 1, 1)),
                np.zeros((bg_patches.shape[0], 1, 1, 1, 1)),
            ],
        )
        if self.weigh_scribbles:
            fg_orig = scribbles_mask[scribbles == self.scribbles_fg_label]
            bg_orig = scribbles_mask[scribbles == self.scribbles_bg_label]

            all_scrib_weights = np.concatenate([fg_orig, bg_orig], axis=0)

            # [ # simulated scribbles, # original scribbles]
            number_of_original_scribbles = [
                (all_scrib_weights == 0).sum(),
                (all_scrib_weights == 1).sum(),
            ]
            eps = get_eps(image)
            weight_for_scribbles = [
                (1.0 / (x + eps))
                * (
                    sum(number_of_original_scribbles)
                    / len(number_of_original_scribbles)
                )
                for x in number_of_original_scribbles
            ]
            logging.info(
                "Simulated vs orig scribbles:{}".format(number_of_original_scribbles)
            )
            logging.info("Weights per type scribbles: {}".format(weight_for_scribbles))

            for i in np.unique(all_scrib_weights):
                all_scrib_weights[all_scrib_weights == int(i)] = weight_for_scribbles[
                    int(i)
                ]

            all_scrib_weights = (
                torch.from_numpy(all_scrib_weights)
                .type(torch.float32)
                .to(device=self.device)
            )

        image_patches_pt = (
            torch.from_numpy(all_sel_patches).type(torch.float32).to(device=self.device)
        )

        target_pt = (
            torch.from_numpy(all_sel_labels).type(torch.long).to(device=self.device)
        )
        logging.info(
            "Training using model features {} and loss {}".format(self.model, self.loss)
        )
        if self.model == "HAAR":
            model = OnlineModelMLPFCNHaarFeatures(
                kernel_size=self.kernel_size,
                hidden_layers=self.hidden_layers,
                num_classes=num_classes,
                haar_padding="valid",
                use_bn=True,
                activation=torch.nn.ReLU,
                dropout=self.dropout,
            ).to(device=self.device)
        elif self.model == "FEAT":
            model = OnlineModelMLPFCNLearnedFeatures(
                feat_kernel_size=self.kernel_size,
                feat_num_filters=self.num_filters,
                feat_num_layers=self.num_layers,
                hidden_layers=self.hidden_layers,
                num_classes=num_classes,
                feat_padding="valid",
                use_bn=True,
                activation=torch.nn.ReLU,
                dropout=self.dropout,
            ).to(device=self.device)
        else:
            raise ValueError("Unknown model specified, {}".format(self.model))

        # if a model checkout found, load params
        if self.model_path and os.path.exists(self.model_path):
            # load
            logging.info("Loading online model from: %s" % self.model_path)
            try:
                # sometimes this may fail, particularly if the model has changed since last saved pt
                # this really is meant to be a temporary copy so okay to delete it in such cases
                model.load_state_dict(torch.load(self.model_path))
            except:
                logging.info(
                    "Unable to load weights, deleting previous model checkpoint at: {}".format(
                        self.model_path
                    )
                )
                os.unlink(self.model_path)

        if self.train_feat:
            params_to_train = [p for n, p in model.named_parameters()]
        else:
            params_to_train = [
                p for n, p in model.named_parameters() if "featureextactor" not in n
            ]
            # if haar is not to be learned, then pre-compute features once and reuse to save compute
            with torch.no_grad():
                image_patches_pt = model(
                    image_patches_pt, skip_feat=False, skip_mlp=True
                )

        optim = torch.optim.Adam(params_to_train, lr=self.lr)
        # optim = torch.optim.RMSprop(params_to_train, lr=self.lr)
        # optim = torch.optim.SGD(params_to_train, lr=self.lr)

        # use multi step at epochs * lr_step epoch to reduce lr by a factor of lr
        # e.g. epoch=50, lr_step=0.5 and lr=0.1, then lr=0.1 for epoch[0-25] and lr=0.01 for epoch[25-50]
        # or use cosine annealing method
        logging.info("Using LR Scheduler {}".format(self.lr_scheduler))
        if self.lr_step and self.lr_scheduler == "MULTI":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim,
                [int(self.epochs * lrstep) for lrstep in list(self.lr_step)],
                gamma=0.1,
            )
        elif self.lr_scheduler == "COSINE":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.epochs
            )
        else:
            raise ValueError(
                "Unrecognised lr_scheduler == {}".format(self.lr_scheduler)
            )

        # calculate imbalance weights for cross-entropy
        # help on imbalanced cross-entropy from:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
        number_of_samples = [np.sum(all_sel_labels == x) for x in range(num_classes)]
        # if even one class missing, then skip weighting
        skip_weighting = 0 in number_of_samples
        if not skip_weighting:
            eps = get_eps(image)
            weight_for_classes = (
                torch.tensor(
                    [
                        (1.0 / (x + eps))
                        * (sum(number_of_samples) / len(number_of_samples))
                        for x in number_of_samples
                    ]
                )
                .type(torch.float32)
                .to(self.device)
            )
        else:
            logging.info(
                "Skipping weighting for class, as atleast one class not in training data"
            )
            weight_for_classes = (
                torch.tensor([1.0] * len(number_of_samples))
                .to(torch.float32)
                .to(self.device)
            )
        logging.info("Samples per class:{}".format(number_of_samples))
        logging.info("Weights per class: {}".format(weight_for_classes))

        if self.weigh_scribbles:
            reduction = "none"
        else:
            reduction = "mean"

        if self.loss == "CE":
            loss_func = torch.nn.CrossEntropyLoss(
                weight=weight_for_classes, ignore_index=-1, reduction=reduction
            )
            target_pt = target_pt.squeeze(1)
        elif self.loss == "DICE":
            loss_func = DiceLoss(to_onehot_y=True, softmax=True, reduction=reduction)
        elif self.loss == "DICECE":
            loss_func = MyDiceCELoss(
                to_onehot_y=True,
                softmax=True,
                ce_weight=weight_for_classes,
                reduction=reduction,
            )
        else:
            raise ValueError("Invalid loss received {}".format(self.loss))

        # amp help from tutorial:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        model.train()
        pbar = tqdm.tqdm(range(self.epochs))
        for ep in pbar:
            # for idx, (patch_data, target_data) in enumerate(loader):
            # patch_data, target_data = patch_data.to(self.device), target_data.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output_pt = model(image_patches_pt, skip_feat=not self.train_feat)
                loss = loss_func(output_pt, target_pt)
                del output_pt

                if self.weigh_scribbles:
                    # give more importance to user-provided scribbles
                    loss = (loss.mean(dim=1) * all_scrib_weights).mean()

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            pbar.set_description("Online Model Loss: %f" % loss.item())

            if loss.item() < self.loss_threshold:
                # stop training if loss criteria met
                break
            if self.lr_step:
                scheduler.step()

        # clear some GPU and CPU memory
        del (
            image_patches_pt,
            image_patches,
            target_pt,
            fg_patches,
            bg_patches,
            all_sel_labels,
            all_sel_patches,
        )
        model.eval()

        # save model to storage if needed
        if self.model_path:
            logging.info("Saving online model to: %s" % self.model_path)
            torch.save(model.state_dict(), self.model_path)

        image_pt = (
            torch.from_numpy(np.expand_dims(np.expand_dims(image, axis=0), axis=0))
            .type(torch.float32)
            .to(device=self.device)
        )

        with torch.no_grad():
            try:
                output_pt = model(image_pt)
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise RuntimeError(e)
                logging.info(str(e))
                logging.info("Not enough memory for online inference")
                logging.info("Trying inference using CPU (slower)")
                model = model.to("cpu")
                output_pt = model(image_pt.to("cpu"))

            output_pt = torch.softmax(output_pt, dim=1)

        if self.use_argmax:
            output_pt = torch.argmax(output_pt, dim=1, keepdim=True).type(torch.float32)

        output_np = output_pt.squeeze(0).detach().cpu().numpy()

        d[self.post_proc_label] = output_np

        return d


class MakeLikelihoodFromScribblesAdaptiveMONetd(MyInteractiveSegmentationTransform):
    """
    Make Likelihood using Adaptive Multi-scale Online Likelihood Network (MONet)
    from paper:

    TODO: Add paper link
    """

    def __init__(
        self,
        image: str,
        scribbles: str,
        adaptive_weights: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "prob",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        model: str = "MSFEAT",
        loss: str = "CE",
        epochs: int = 80,
        lr: float = 0.1,
        lr_step: float = [0.1],
        lr_scheduler: str = "COSINE",
        dropout: float = 0.2,
        hidden_layers: List[int] = [16, 8],
        kernel_size: int = 9,
        num_filters: int = 128,
        num_layers: int = 1,
        scales: int = [1, 3, 5, 9],
        train_feat: bool = True,
        use_argmax: bool = False,
        model_path: str = None,
        loss_threshold: float = 0.0,  # 0.005,
        use_amp: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__(meta_key_postfix)
        self.image = image
        self.scribbles = scribbles
        self.adaptive_weights = adaptive_weights
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.post_proc_label = post_proc_label
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.lr_step = lr_step
        self.lr_scheduler = lr_scheduler
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.hidden_layers = hidden_layers
        self.scales = scales
        self.train_feat = train_feat
        self.use_argmax = use_argmax
        self.model_path = model_path
        self.loss_threshold = loss_threshold
        self.use_amp = use_amp
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # at the moment only works for binary seg problem
        num_classes = 2

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.post_proc_label)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        scribbles = self._fetch_data(d, self.scribbles)
        adaptive_weights = self._fetch_data(d, self.adaptive_weights)

        if self.scribbles + "_orig" in d.keys():
            scribbles_mask = np.squeeze(self._fetch_data(d, self.scribbles + "_orig"))
            scribbles_mask = (scribbles_mask == self.scribbles_bg_label) | (
                scribbles_mask == self.scribbles_fg_label
            )
            scribbles_mask = scribbles_mask.astype(np.float32)
        else:
            raise ValueError(
                "No original scribbles ({}_orig) volume found".format(self.scribbles)
            )

        image = np.squeeze(image)
        scribbles = np.squeeze(scribbles)
        adaptive_weights = np.squeeze(adaptive_weights)

        # zero-pad input image volume
        pad_size = int(math.floor(self.kernel_size / 2))
        image = np.pad(image, ((pad_size, pad_size),) * 3, mode="symmetric")

        # extract patches and select only relevant patches with scribble labels for online training
        image_patches = view_as_windows(
            image,
            (self.kernel_size, self.kernel_size, self.kernel_size),
            step=1,
        )

        # select relevant patches only for training network
        fg_patches = image_patches[scribbles == self.scribbles_fg_label]
        bg_patches = image_patches[scribbles == self.scribbles_bg_label]

        fg_ad_weights = np.max(adaptive_weights, axis=0)
        fg_ad_weights = fg_ad_weights[scribbles == self.scribbles_fg_label]
        bg_ad_weights = np.max(adaptive_weights, axis=0)
        bg_ad_weights = bg_ad_weights[scribbles == self.scribbles_bg_label]

        fg_orig_mask = scribbles_mask[scribbles == self.scribbles_fg_label]
        bg_orig_mask = scribbles_mask[scribbles == self.scribbles_bg_label]

        all_sel_patches = np.expand_dims(
            np.concatenate([fg_patches, bg_patches], axis=0), axis=1
        )
        all_sel_labels = np.concatenate(
            [
                np.ones((fg_patches.shape[0], 1, 1, 1, 1)),
                np.zeros((bg_patches.shape[0], 1, 1, 1, 1)),
            ],
        )

        # intra-scribbles
        eps = get_eps(image)

        number_of_original_scribbles = [fg_orig_mask.sum(), bg_orig_mask.sum()]
        logging.info(
            "number of original scribbles: {}".format(number_of_original_scribbles)
        )
        number_of_support_scribbles = [
            x - y
            for x, y in zip(
                [fg_orig_mask.size, bg_orig_mask.size],
                number_of_original_scribbles,
            )
        ]
        logging.info(
            "number of support scribbles: {}".format(number_of_support_scribbles)
        )

        self.weighting_method = "effectivecount"
        logging.info("Using weighting method {}".format(self.weighting_method))

        if 0 in number_of_original_scribbles:
            logging.info("Missing at least one label, falling back to default numbers")
            number_of_original_scribbles = [
                max(number_of_original_scribbles + [100.0])
            ] * len(number_of_original_scribbles)

        if 0 in number_of_support_scribbles:
            logging.info("Missing at least one label, falling back to default numbers")
            number_of_support_scribbles = [
                max(number_of_support_scribbles + [100.0])
            ] * len(number_of_support_scribbles)

        if self.weighting_method == "effectivecount":
            # https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
            B = 0.99995
            effective_number_of_original_scribbles = [
                (1 - B**x) / (1 - B) for x in number_of_original_scribbles
            ]
            effective_number_of_support_scribbles = [
                (1 - B**x) / (1 - B) for x in number_of_support_scribbles
            ]
            logging.info(
                "Effective number of original scribbles {}".format(
                    effective_number_of_original_scribbles
                )
            )
            logging.info(
                "Effective number of support scribbles {}".format(
                    effective_number_of_support_scribbles
                )
            )

        if self.weighting_method == "exp":
            weight_for_scribbles = [
                math.exp(
                    -(
                        x
                        / (
                            sum(number_of_original_scribbles)
                            + sum(number_of_support_scribbles)
                            + eps
                        )
                    )
                )
                for x in number_of_original_scribbles
            ]
        elif self.weighting_method == "noweight":
            weight_for_scribbles = [1.0 for x in number_of_original_scribbles]
        elif self.weighting_method == "depline":
            weight_for_scribbles = [
                (1.0 / (x + eps))
                * (
                    (
                        sum(number_of_original_scribbles)
                        + sum(number_of_support_scribbles)
                    )
                    / (len(number_of_original_scribbles))
                )
                for x in number_of_original_scribbles
            ]
        elif self.weighting_method == "effectivecount":
            # # https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
            weight_for_scribbles = [
                (1.0 / (x + eps))
                * (
                    (
                        sum(effective_number_of_original_scribbles)
                        + sum(effective_number_of_support_scribbles)
                    )
                    / (
                        len(effective_number_of_original_scribbles)
                        + len(effective_number_of_support_scribbles)
                    )
                )
                for x in effective_number_of_original_scribbles
            ]
        else:
            raise ValueError(
                "invalid weighting strategy [{}] selected".format(self.weighting_method)
            )

        if self.weighting_method == "exp":
            weight_for_support = [
                math.exp(
                    -(
                        x
                        / (
                            sum(number_of_support_scribbles)
                            + sum(number_of_original_scribbles)
                            + eps
                        )
                    )
                )
                for x in number_of_support_scribbles
            ]
        elif self.weighting_method == "noweight":
            weight_for_support = [1.0 for x in number_of_support_scribbles]
        elif self.weighting_method == "depline":
            weight_for_support = [
                (1.0 / (x + eps))
                * (
                    (
                        sum(number_of_support_scribbles)
                        + sum(number_of_original_scribbles)
                    )
                    / (len(number_of_support_scribbles))
                )
                for x in number_of_support_scribbles
            ]
        elif self.weighting_method == "effectivecount":
            weight_for_support = [
                (1.0 / (x + eps))
                * (
                    (
                        sum(effective_number_of_support_scribbles)
                        + sum(effective_number_of_original_scribbles)
                    )
                    / (
                        len(effective_number_of_support_scribbles)
                        + len(effective_number_of_original_scribbles)
                    )
                )
                for x in effective_number_of_support_scribbles
            ]
        else:
            raise ValueError(
                "invalid weighting strategy [{}] selected".format(self.weighting_method)
            )

        logging.info("weight scribbles: {}".format(weight_for_scribbles))
        logging.info("weight support: {}".format(weight_for_support))

        fg_weights = (
            fg_ad_weights * weight_for_scribbles[0]
            + (1 - fg_ad_weights) * weight_for_support[0]
        )
        bg_weights = (
            bg_ad_weights * weight_for_scribbles[1]
            + (1 - bg_ad_weights) * weight_for_support[1]
        )
        all_sel_weights = np.concatenate([fg_weights, bg_weights], axis=0)

        image_patches_pt = (
            torch.from_numpy(all_sel_patches).type(torch.float32).to(device=self.device)
        )

        target_pt = (
            torch.from_numpy(all_sel_labels).type(torch.long).to(device=self.device)
        )

        weights_pt = (
            (
                torch.from_numpy(all_sel_weights)
                .type(torch.float32)
                .to(device=self.device)
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        logging.info(
            "Training using model features {} and loss {}".format(self.model, self.loss)
        )
        if self.model == "HAAR":
            model = OnlineModelMLPFCNHaarFeatures(
                kernel_size=self.kernel_size,
                hidden_layers=self.hidden_layers,
                num_classes=num_classes,
                haar_padding="valid",
                use_bn=True,
                activation=torch.nn.ReLU,
                dropout=self.dropout,
            ).to(device=self.device)
        elif self.model == "FEAT":
            model = OnlineModelMLPFCNLearnedFeatures(
                feat_kernel_size=self.kernel_size,
                feat_num_filters=self.num_filters,
                feat_num_layers=self.num_layers,
                hidden_layers=self.hidden_layers,
                num_classes=num_classes,
                feat_padding="valid",
                use_bn=True,
                activation=torch.nn.ReLU,
                dropout=self.dropout,
            ).to(device=self.device)
        elif self.model == "MSFEAT":
            model = OnlineModelMLPMultiScaleFCNLearnedFeatures(
                feat_kernel_size=self.kernel_size,
                feat_num_filters=self.num_filters,
                feat_num_layers=self.num_layers,
                hidden_layers=self.hidden_layers,
                num_classes=num_classes,
                feat_padding="valid",
                use_bn=True,
                activation=torch.nn.ReLU,
                dropout=self.dropout,
                scales=self.scales,
            ).to(device=self.device)
        else:
            raise ValueError("Unknown model specified, {}".format(self.model))

        # if a model checkout found, load params
        if self.model_path and os.path.exists(self.model_path):
            # load
            logging.info("Loading online model from: %s" % self.model_path)
            try:
                # sometimes this may fail, particularly if the model has changed since last saved pt
                # this really is meant to be a temporary copy so okay to delete it in such cases
                model.load_state_dict(torch.load(self.model_path))
            except:
                logging.info(
                    "Unable to load weights, deleting previous model checkpoint at: {}".format(
                        self.model_path
                    )
                )
                os.unlink(self.model_path)

        if self.train_feat:
            params_to_train = [p for n, p in model.named_parameters()]
        else:
            params_to_train = [
                p for n, p in model.named_parameters() if "featureextactor" not in n
            ]
            with torch.no_grad():
                image_patches_pt = model(
                    image_patches_pt, skip_feat=False, skip_mlp=True
                )

        optim = torch.optim.Adam(params_to_train, lr=self.lr)
        # optim = torch.optim.RMSprop(params_to_train, lr=self.lr)
        # optim = torch.optim.SGD(params_to_train, lr=self.lr)
        # optim = torch.optim.LBFGS(params_to_train, lr=self.lr)

        # use multi step at epochs * lr_step epoch to reduce lr by a factor of lr
        # e.g. epoch=50, lr_step=0.5 and lr=0.1, then lr=0.1 for epoch[0-25] and lr=0.01 for epoch[25-50]
        # or use cosine annealing method
        logging.info("Using LR Scheduler {}".format(self.lr_scheduler))
        if self.lr_step and self.lr_scheduler == "MULTI":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim,
                [int(self.epochs * lrstep) for lrstep in list(self.lr_step)],
                gamma=0.1,
            )
        elif self.lr_scheduler == "COSINE":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.epochs
            )
        else:
            raise ValueError(
                "Unrecognised lr_scheduler == {}".format(self.lr_scheduler)
            )

        reduction = "none"
        if self.loss == "CE":
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)
            target_pt = target_pt.squeeze(1)
        elif self.loss == "DICE":
            loss_func = DiceLoss(to_onehot_y=True, softmax=True, reduction=reduction)
        elif self.loss == "DICECE":
            loss_func = MyDiceCELoss(
                to_onehot_y=True,
                softmax=True,
                reduction=reduction,
            )
        else:
            raise ValueError("Invalid loss received {}".format(self.loss))

        # amp help from tutorial:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        model.train()
        pbar = tqdm.tqdm(range(self.epochs))
        for ep in pbar:
            # for idx, (patch_data, target_data) in enumerate(loader):
            # patch_data, target_data = patch_data.to(self.device), target_data.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output_pt = model(image_patches_pt, skip_feat=not self.train_feat)
                loss = loss_func(output_pt, target_pt)
                del output_pt

                # if self.weigh_scribbles:
                # give more importance to user-provided scribbles
                loss = (loss * weights_pt).mean()

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            pbar.set_description("Online Model Loss: %f" % loss.item())

            if loss.item() < self.loss_threshold:
                # stop training if loss criteria met
                break
            if self.lr_step:
                scheduler.step()

        # clear some GPU and CPU memory
        del (
            image_patches_pt,
            image_patches,
            target_pt,
            fg_patches,
            bg_patches,
            all_sel_labels,
            all_sel_patches,
        )
        model.eval()

        # save model to storage if needed
        if self.model_path:
            logging.info("Saving online model to: %s" % self.model_path)
            torch.save(model.state_dict(), self.model_path)

        image_pt = (
            torch.from_numpy(np.expand_dims(np.expand_dims(image, axis=0), axis=0))
            .type(torch.float32)
            .to(device=self.device)
        )

        with torch.no_grad():
            try:
                output_pt = model(image_pt)
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise RuntimeError(e)
                logging.info(str(e))
                logging.info("Not enough memory for online inference")
                logging.info("Trying inference using CPU (slower)")
                model = model.to("cpu")
                output_pt = model(image_pt.to("cpu"))

            output_pt = torch.softmax(output_pt, dim=1)

        if self.use_argmax:
            output_pt = torch.argmax(output_pt, dim=1, keepdim=True).type(torch.float32)

        output_np = output_pt.squeeze(0).detach().cpu().numpy()

        d[self.post_proc_label] = output_np

        return d


class ApplyProbGuidedDropoutLabeld(MyInteractiveSegmentationTransform):
    """
    Apply Probability Guided pruning of labels using
    probability from initial segmentation and a dropout factor
    as defined in:

    TODO: Add paper link

    """

    def __init__(
        self,
        scribbles: str,
        logits: str,
        confidence: float = 0.85,
        fg_dropout: float = 0.0,
        bg_dropout: float = 0.0,
        meta_key_postfix: str = "meta_dict",
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        save_orig_scribbles: bool = True,
    ):
        super().__init__(meta_key_postfix)
        self.scribbles = scribbles
        self.logits = logits
        self.confidence = confidence
        self.fg_dropout = fg_dropout
        self.bg_dropout = bg_dropout
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.save_orig_scribbles = save_orig_scribbles

    def __call__(self, data):
        d = dict(data)

        scribbles = self._fetch_data(d, self.scribbles)
        logits = self._fetch_data(d, self.logits).copy()
        logits = self._normalise_logits(logits)
        predlabel = np.expand_dims(np.argmax(logits, axis=0), axis=0)
        # logits = np.max(logits, axis=0, keepdims=True)
        uncertainity_logits = 1 - make_uncertainity_from_logits(
            logits, axis=0
        )  # make certainity

        if predlabel.shape[0] > 1:
            raise ValueError("Only supports argmax label output")

        predlabel[predlabel == 0] = (
            (predlabel == 0).astype(np.float32) * self.scribbles_bg_label
        )[predlabel == 0]
        predlabel[predlabel == 1] = (
            (predlabel == 1).astype(np.float32) * self.scribbles_fg_label
        )[predlabel == 1]

        # for a very dense background, it helps to dropout some of the samples
        # especially for the online learning methods
        if (
            self.fg_dropout > 0.0
            and self.fg_dropout < 1.0
            and self.bg_dropout > 0.0
            and self.bg_dropout < 1.0
        ):
            conf_mask = create_drop_prob_mask(
                uncertainity_logits, confidence=self.confidence, inequality_less=False
            )
            fg_mask = (
                create_drop_value_mask(predlabel.shape, dropout=self.fg_dropout)
                * conf_mask
            )
            bg_mask = (
                create_drop_value_mask(predlabel.shape, dropout=self.bg_dropout)
                * conf_mask
            )
            mask = fg_mask * (predlabel == self.scribbles_fg_label) + bg_mask * (
                predlabel == self.scribbles_bg_label
            )
            predlabel = predlabel * mask
        else:
            logging.info(
                "Warning: unable to apply dropout for fg: {} bg: {}".format(
                    self.fg_dropout, self.bg_dropout
                )
            )

        # Add user scribbles as dense scribble labels
        predlabel[scribbles == self.scribbles_bg_label] = self.scribbles_bg_label
        predlabel[scribbles == self.scribbles_fg_label] = self.scribbles_fg_label

        if 1 == 0:
            predlabel = (predlabel > 0).astype(np.float32)

        d[self.scribbles] = predlabel

        if self.save_orig_scribbles:
            d[self.scribbles + "_orig"] = scribbles

        return d


class MakeEGDWeights(MyInteractiveSegmentationTransform):
    """
    Make Exponential Geodesic Distance based weights W
    with additional temperature Tau term from paper:

    TODO: Add paper link

    """

    def __init__(
        self,
        image: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        egd_weights: str = "egd_weights",
        tau: float = 1.0,
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
        use_fastgeodis: bool = False,
        renormalise_image: bool = True,
        normalise_prob: bool = True,
        hard_weights: bool = False,
        device: str = "cuda",
    ) -> None:
        super(MakeEGDWeights, self).__init__(meta_key_postfix)
        self.image = image
        self.scribbles = scribbles
        self.egd_weights = egd_weights
        self.tau = tau
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.use_fastgeodis = use_fastgeodis
        self.renormalise_image = renormalise_image
        self.normalise_prob = normalise_prob
        self.hard_weights = hard_weights
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # attempt to fetch algorithmic parameters from app if present
        self.tau = d.get("tau", self.tau)

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.egd_weights)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        scribbles = self._fetch_data(d, self.scribbles)

        if self.renormalise_image:
            image = self._normalise_range(image, b_min=0.0, b_max=1.0)

        spacing = self._get_spacing(d, self.image)

        egd_weights_term = make_egd_adaptive_weights(
            image=image,
            scribbles=scribbles,
            scribbles_fg_label=self.scribbles_fg_label,
            scribbles_bg_label=self.scribbles_bg_label,
            spacing=spacing,
            tau=self.tau,
            use_fastgeodis=self.use_fastgeodis,
            device=self.device,
        )

        if self.hard_weights:
            egd_weights_term = (
                (egd_weights_term[[0], ...] > 0.5) | (egd_weights_term[[1], ...] > 0.5)
            ).astype(np.float32)

        if self.normalise_prob:
            egd_weights_term = self._normalise_logits(egd_weights_term, 0)

        d[self.egd_weights] = egd_weights_term

        return d


class CombineCNNAndLikelihoodEGDd(MyInteractiveSegmentationTransform):
    """
    Combine Likelihood output with CNN segmentation
    """

    def __init__(
        self,
        distance_term: str,
        likelihood: str,
        logits: str,
        meta_key_postfix: str = "meta_dict",
        softmask_label: str = "prob",
        normalise_terms: bool = True,
    ) -> None:
        super().__init__(meta_key_postfix)
        self.distance_term = distance_term
        self.likelihood = likelihood
        self.logits = logits
        self.softmask_label = softmask_label
        self.normalise_terms = normalise_terms

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.logits, dst=self.softmask_label)

        # read relevant terms from data
        distance_term = self._fetch_data(d, self.distance_term)
        likelihood = self._fetch_data(d, self.likelihood)
        logits = self._fetch_data(d, self.logits)

        if self.normalise_terms:
            # no need to normalise distance_term - it is not a probability
            # distance_term = self._normalise_logits(distance_term, axis=0)
            likelihood = self._normalise_logits(likelihood, axis=0)
            logits = self._normalise_logits(logits, axis=0)

        # make likelihood image
        alpha_i = np.max(distance_term, axis=0, keepdims=True)
        softmask_label = alpha_i * likelihood + (1 - alpha_i) * logits

        d[self.softmask_label] = softmask_label

        return d


class RoundArrayd(MapTransform):
    """
    Round input image to nearest integer
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = np.round(d[key])
        return d


class ApplyGaussianSmoothing(MyInteractiveSegmentationTransform):
    """
    Apply Gaussian Smoothing transform.
    Applies Gaussian filter to input image volume, used mainly to filter noise
    """

    def __init__(
        self,
        image: str,
        meta_key_postfix="meta_dict",
        kernel_size: int = 3,
        sigma: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__(meta_key_postfix)
        self.image = image
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device

    def __call__(self, data):
        d = dict(data)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)

        # determine dimensionality of input tensor
        spatial_dims = len(image.shape) - 1

        # add batch dimension
        image = torch.from_numpy(image).to(self.device).unsqueeze_(0)

        # initialise smoother
        if spatial_dims == 2:
            smoother = GaussianFilter2d(
                in_channels=image.shape[1],
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                padding="same",
                stride=1,
                padding_mode="zeros",
            ).to(self.device)
        elif spatial_dims == 3:
            smoother = GaussianFilter3d(
                in_channels=image.shape[1],
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                padding="same",
                stride=1,
                padding_mode="zeros",
            ).to(self.device)
        else:
            raise ValueError(
                "Gaussian smoothing not defined for {}-dimensional input".format(
                    spatial_dims
                )
            )

        # apply smoothing
        image = smoother(image).squeeze(0).detach().cpu().numpy()

        d[self.image] = image

        return d


class ApplyGraphCutOptimisationd(InteractiveSegmentationTransform):
    """
    GraphCut optimisation transform.
    This can be used in conjuction with any Make*Unaryd transform
    (e.g. MakeISegUnaryd from above for implementing ISeg unary term).
    It optimises a typical energy function for interactive segmentation methods using numpymaxflow's GraphCut method,
    e.g. Equation 5 from https://arxiv.org/pdf/1710.04043.pdf.
    Usage Example::
        Compose(
            [
                # unary term maker
                MakeISegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                ),
                # optimiser
                ApplyGraphCutOptimisationd(
                    unary="unary",
                    pairwise="image",
                    post_proc_label="pred",
                    lamda=10.0,
                    sigma=15.0,
                ),
            ]
        )
    """

    def __init__(
        self,
        unary: str,
        pairwise: str,
        meta_key_postfix: str = "meta_dict",
        post_proc_label: str = "pred",
        lamda: float = 8.0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(meta_key_postfix)
        self.unary = unary
        self.pairwise = pairwise
        self.post_proc_label = post_proc_label
        self.lamda = lamda
        self.sigma = sigma

    def __call__(self, data):
        d = dict(data)

        # attempt to fetch algorithmic parameters from app if present
        self.lamda = d.get("lamda", self.lamda)
        self.sigma = d.get("sigma", self.sigma)

        # copy affine meta data from pairwise input
        self._copy_affine(d, self.pairwise, self.post_proc_label)

        # read relevant terms from data
        unary_term = self._fetch_data(d, self.unary)
        pairwise_term = self._fetch_data(d, self.pairwise)

        # check if input unary is compatible with GraphCut opt
        if unary_term.shape[0] > 2:
            raise ValueError(
                "GraphCut can only be applied to binary probabilities, received {}".format(
                    unary_term.shape[0]
                )
            )

        # # attempt to unfold probability term
        # unary_term = self._unfold_prob(unary_term, axis=0)

        # prepare data for numpymaxflow's GraphCut
        # run GraphCut
        post_proc_label = maxflow(
            pairwise_term, unary_term, lamda=self.lamda, sigma=self.sigma
        )

        d[self.post_proc_label] = post_proc_label

        return d
