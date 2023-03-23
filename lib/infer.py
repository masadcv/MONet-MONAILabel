from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CopyItemsd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    ToNumpyd,
    ToTensord,
)
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.scribbles.transforms import WriteLogits
from monailabel.transform.post import BoundingBoxd, Restored


class InferCovid(InferTask):
    """
    This provides Inference Engine for pre-trained Lung Lesion segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation of the lung lesion from CT volume",
        intensity_range=(-1000, 500, 0.0, 1.0, True),
        pix_dim=(1.25, 1.25, 5.0),
    ):
        labels = "lesion"
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )
        self.intensity_range = intensity_range
        self.pix_dim = pix_dim

    def pre_transforms(self):
        return [
            LoadImaged(keys="image"),
            AddChanneld(keys="image"),
            Spacingd(keys="image", pixdim=self.pix_dim),
            ScaleIntensityRanged(
                keys="image",
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[2],
                b_max=self.intensity_range[3],
                clip=self.intensity_range[4],
            ),
            ToTensord(keys="image"),
        ]

    def inferer(self):
        return SlidingWindowInferer(roi_size=[192, 192, 16])

    def post_transforms(self):
        return [
            CopyItemsd(keys="pred", times=1, names="logits"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys=["pred", "logits"]),
            Restored(keys=["pred", "logits"], ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
            WriteLogits(key="logits", result="result"),
        ]
