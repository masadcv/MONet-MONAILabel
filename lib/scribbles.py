import logging

logger = logging.getLogger(__name__)

from monai.transforms import (
    AsDiscreted,
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.scribbles.transforms import MakeISegUnaryd
from monailabel.transform.post import BoundingBoxd, Restored
from monailabel.transform.writer import Writer

from lib.transforms import (
    ApplyGaussianSmoothing,
    ApplyGraphCutOptimisationd,
    ApplyProbGuidedDropoutLabeld,
    # CombineCNNAndLikelihoodEGDd,
    MakeEGDWeights,
    MakeLikelihoodFromScribblesAdaptiveMONetd,
    MakeLikelihoodFromScribblesECONetd,
    RoundArrayd,
)


class MyInferTask(InferTask):
    def writer(self, data, extension=None, dtype=None):
        """
        Override Writer to using nibabel for storing labels
        """
        logger.info("Writing Result")
        if extension is not None:
            data["result_extension"] = extension
        if dtype is not None:
            data["result_dtype"] = dtype

        writer = Writer(
            label=self.output_label_key, json=self.output_json_key, nibabel=True
        )
        return writer(data)


class LikelihoodBasedSegmentation(MyInferTask):
    def __init__(
        self,
        dimension=3,
        description="Generic base class for constructing online likelihood based segmentors",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=2.5,
        sigma=0.15,
        scribbles_bg_label=2,
        scribbles_fg_label=3,
        apply_graphcut=True,
        config=None,
    ):
        labels = "lesion"
        params = {"lamda": lamda, "sigma": sigma}
        if config:
            config.update(params)
        else:
            config = params

        super().__init__(
            path=None,
            network=None,
            labels=labels,
            type=InferType.SCRIBBLES,
            dimension=dimension,
            description=description,
            config=config,
        )
        self.intensity_range = intensity_range
        self.pix_dim = pix_dim
        self.lamda = lamda
        self.sigma = sigma
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label
        self.apply_graphcut = apply_graphcut

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "label", "logits"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(
                keys=["image", "logits", "label"],
                pixdim=self.pix_dim,
                mode=["bilinear", "bilinear", "nearest"],
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[2],
                b_max=self.intensity_range[3],
                clip=self.intensity_range[4],
            ),
        ]

    def post_transforms(self):
        if self.apply_graphcut:
            init_tx = [
                # optimiser
                ApplyGraphCutOptimisationd(
                    unary="prob",
                    pairwise="image",
                    post_proc_label="pred",
                    lamda=self.lamda,
                    sigma=self.sigma,
                ),
            ]
        else:
            init_tx = [
                CopyItemsd(keys="prob", times=1, names="pred"),
                CopyItemsd(keys="image_meta_dict", times=1, names="pred_meta_dict"),
            ]

        return init_tx + [
            Restored(keys="pred", ref_image="image", mode="trilinear"),
            RoundArrayd(keys="pred"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]


class OnlineAdaptiveMONetBasedGraphCut(LikelihoodBasedSegmentation):
    """
    Defines interacgtive segmentation using Multi-scale Online Likelihood Network (MONet) online training and inference
    for COVID-19 lung lesion segmentation based on the following paper:

    TODO: Add paper details

    This task takes as input 1) original image volume, 2) initial segmentation by a CNN, and 3) scribbles from user
    indicating foreground and background regions. A likelihood volume is learned and inferred using MONet using
    adaptive online training and probability guided pruning proposed in paper above.

    numpymaxflow's GraphCut layer is used to regularise the resulting likelihood, where unaries come from likelihood
    and pairwise is the original input volume.

    This also implements variations of MONet without multi-scale features, referred as MONet-NoMS in the paper.
    """

    def __init__(
        self,
        dimension=3,
        description="Interactive segmentation with MONet/MONet-NoMS for COVID-19 lung lesion segmentation",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=2.5,
        sigma=0.15,
        tau=0.3,
        model="MSFEAT",
        loss="CE",
        epochs=200,
        lr=0.01,
        lr_step=[0.9, 0.95],
        lr_scheduler="COSINE",
        dropout=0.3,
        hidden_layers=[32, 16],
        num_layers=1,
        kernel_size=9,
        num_filters=128,
        scales=[1, 3, 5, 9],
        train_feat=True,
        model_path=None,
        config=None,
    ):
        super().__init__(
            dimension=dimension,
            description=description,
            intensity_range=intensity_range,
            pix_dim=pix_dim,
            lamda=lamda,
            sigma=sigma,
            scribbles_bg_label=2,
            scribbles_fg_label=3,
            config=config,
        )
        self.tau = tau
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.lr_step = lr_step
        self.lr_scheduler = lr_scheduler
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.scales = scales
        self.train_feat = train_feat
        self.model_path = model_path

    def inferer(self):
        logging.info("Using Fixed Tau: {}".format(self.tau))
        return Compose(
            [
                CopyItemsd(keys="logits", times=1, names="logits_orig"),
                AsDiscreted(keys="logits", argmax=True),
                ApplyProbGuidedDropoutLabeld(
                    scribbles="label",
                    logits="logits_orig",
                    confidence=0.8,
                    fg_dropout=0.98,
                    bg_dropout=0.98,
                    scribbles_bg_label=self.scribbles_bg_label,
                    scribbles_fg_label=self.scribbles_fg_label,
                    save_orig_scribbles=True,
                ),
                MakeEGDWeights(
                    image="image",
                    scribbles="label" + "_orig",  # needs original scribbles
                    egd_weights="egd_weights",
                    tau=self.tau,
                    scribbles_bg_label=self.scribbles_bg_label,
                    scribbles_fg_label=self.scribbles_fg_label,
                    use_fastgeodis=True,
                    renormalise_image=True,
                    normalise_prob=False,
                    hard_weights=False,
                ),
                MakeLikelihoodFromScribblesAdaptiveMONetd(
                    image="image",
                    scribbles="label",
                    adaptive_weights="egd_weights",
                    post_proc_label="prob",
                    scribbles_bg_label=self.scribbles_bg_label,
                    scribbles_fg_label=self.scribbles_fg_label,
                    model=self.model,
                    loss=self.loss,
                    epochs=self.epochs,
                    lr=self.lr,
                    lr_step=self.lr_step,
                    lr_scheduler=self.lr_scheduler,
                    dropout=self.dropout,
                    hidden_layers=self.hidden_layers,
                    kernel_size=self.kernel_size,
                    num_filters=self.num_filters,
                    num_layers=self.num_layers,
                    scales=self.scales,
                    train_feat=self.train_feat,
                    use_argmax=False,
                    model_path=self.model_path,
                    loss_threshold=0.005,
                    use_amp=False,
                    device="cuda",
                ),
                ApplyGaussianSmoothing(
                    image="image",
                    kernel_size=3,
                    sigma=1.0,
                    device="cuda",
                ),
                # CombineCNNAndLikelihoodEGDd(
                #     distance_term="egd_weights",
                #     likelihood="prob",
                #     logits="logits_orig",
                #     softmask_label="prob",
                #     normalise_terms=True,
                # ),
                # MakeISegUnaryd(
                #     image="image",
                #     logits="prob",
                #     scribbles="label" + "_orig",  # needs original scribbles
                #     unary="prob",
                #     scribbles_bg_label=self.scribbles_bg_label,
                #     scribbles_fg_label=self.scribbles_fg_label,
                # ),
            ]
        )


class InteractiveGraphCut(LikelihoodBasedSegmentation):
    """
    Defines interactive segmentation using interactive Graphcut for
    COVID-19 lung lesion segmentation based on the following paper:

    TODO: Add paper details

    This task takes as input 1) original image volume, 2) initial segmentation by a CNN, and 3) scribbles from user
    indicating foreground and background regions. A refined segmentation using scribbles wit Graphcut as descibed in
    paper above.

    numpymaxflow's GraphCut layer is used to regularise the resulting likelihood, where unaries come from likelihood
    and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="Interactive segmentation with interactive Graphcut for COVID-19 lung lesion segmentation",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=2.5,
        sigma=0.15,
        config=None,
    ):
        super().__init__(
            dimension=dimension,
            description=description,
            intensity_range=intensity_range,
            pix_dim=pix_dim,
            lamda=lamda,
            sigma=sigma,
            scribbles_bg_label=2,
            scribbles_fg_label=3,
            config=config,
        )

    def inferer(self):
        return Compose(
            [
                MakeISegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="prob",
                    scribbles_bg_label=self.scribbles_bg_label,
                    scribbles_fg_label=self.scribbles_fg_label,
                ),
            ]
        )


class OnlineECONetBasedGraphCut(LikelihoodBasedSegmentation):
    """
    Defines Efficient Convolutional Online Likelihood Network (ECONet) based Online Likelihood training and inference method for
    COVID-19 lung lesion segmentation based on the following paper:

    Asad, Muhammad, Lucas Fidon, and Tom Vercauteren. "" ECONet: Efficient Convolutional Online Likelihood Network
    for Scribble-based Interactive Segmentation." MIDL 2022 (preprint: https://arxiv.org/pdf/2201.04584.pdf).
    This task takes as input 1) original image volume and 2) scribbles from user
    indicating foreground and background regions. A likelihood volume is learned and inferred using ECONet method.

    numpymaxflow's GraphCut layer is used to regularise the resulting likelihood, where unaries come from likelihood
    and pairwise is the original input volume.

    This also implements variations of ECONet with hand-crafted features, referred as ECONet-Haar-Like in the paper.
    """

    def __init__(
        self,
        dimension=3,
        description="Online likelihood generation using ECONet for COVID-19 lung lesion segmentation",
        intensity_range=(-1000, 400, 0.0, 1.0, True),
        pix_dim=(2.0, 2.0, 2.0),
        lamda=2.5,
        sigma=0.15,
        model="FEAT",
        loss="CE",
        epochs=100,
        lr=0.01,
        lr_step=[0.7],
        lr_scheduler="COSINE",
        dropout=0.3,
        hidden_layers=[32, 16],
        num_layers=1,
        kernel_size=9,
        num_filters=128,
        train_feat=True,
        model_path=None,
        config=None,
    ):
        super().__init__(
            dimension=dimension,
            description=description,
            intensity_range=intensity_range,
            pix_dim=pix_dim,
            lamda=lamda,
            sigma=sigma,
            scribbles_bg_label=2,
            scribbles_fg_label=3,
            config=config,
        )
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.lr_step = lr_step
        self.lr_scheduler = lr_scheduler
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.train_feat = train_feat
        self.model_path = model_path

    def inferer(self):
        return Compose(
            [
                MakeLikelihoodFromScribblesECONetd(
                    image="image",
                    scribbles="label",
                    post_proc_label="prob",
                    scribbles_bg_label=self.scribbles_bg_label,
                    scribbles_fg_label=self.scribbles_fg_label,
                    weigh_scribbles=False,
                    model=self.model,
                    loss=self.loss,
                    epochs=self.epochs,
                    lr=self.lr,
                    lr_step=self.lr_step,
                    lr_scheduler=self.lr_scheduler,
                    dropout=self.dropout,
                    hidden_layers=self.hidden_layers,
                    kernel_size=self.kernel_size,
                    num_filters=self.num_filters,
                    num_layers=self.num_layers,
                    train_feat=self.train_feat,
                    use_argmax=False,
                    model_path=self.model_path,
                    loss_threshold=0.005,
                    use_amp=False,
                    device="cuda",
                ),
                ApplyGaussianSmoothing(
                    image="image",
                    kernel_size=3,
                    sigma=1.0,
                    device="cuda",
                ),
            ]
        )
