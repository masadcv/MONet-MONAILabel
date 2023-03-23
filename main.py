import logging
import os
import shutil
from typing import Dict

from monai.networks.nets import BasicUNet
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.tasks.activelearning.random import Random

from lib.infer import InferCovid
from lib.scribbles import (
    InteractiveGraphCut,
    OnlineAdaptiveMONetBasedGraphCut,
    OnlineECONetBasedGraphCut,
)

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.network_covid = BasicUNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
        )
        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model_covid = os.path.join(
            self.model_dir, "pretrained_covid.pt"
        )
        if not os.path.exists(self.pretrained_model_covid):
            raise IOError(
                "COVID segmentation model not found at {}\n"
                "Please download and place in above location".format(
                    self.pretrained_model_covid
                )
            )
        self.final_model_covid = os.path.join(self.model_dir, "model_covid.pt")

        self.singlescale_econet_pretrained_path = (
            "./model/econet_singlescale_model=[72.44].pt"
        )
        if not os.path.exists(self.singlescale_econet_pretrained_path):
            raise IOError(
                "Missing pretrained files for online model {}".format(
                    self.singlescale_econet_pretrained_path
                )
            )
        self.current_ss_econet_model_path = "./model/ss_econet_current_sample_temp.pt"
        if os.path.exists(self.current_ss_econet_model_path):
            logging.info(
                "clearing old online model at: {}".format(
                    self.current_ss_econet_model_path
                )
            )
            os.unlink(self.current_ss_econet_model_path)
            shutil.copy(
                self.singlescale_econet_pretrained_path,
                self.current_ss_econet_model_path,
            )

        self.multiscale_monet_pretrained_path = (
            "./model/econet_multiscale_model=[78.53].pt"
        )
        if not os.path.exists(self.multiscale_monet_pretrained_path):
            raise IOError(
                "Missing pretrained files for online model {}".format(
                    self.multiscale_monet_pretrained_path
                )
            )
        self.current_ms_monet_model_path = "./model/ms_monet_current_sample_temp.pt"
        if os.path.exists(self.current_ms_monet_model_path):
            logging.info(
                "clearing old online model at: {}".format(
                    self.current_ms_monet_model_path
                )
            )
            os.unlink(self.current_ms_monet_model_path)
            shutil.copy(
                self.multiscale_monet_pretrained_path, self.current_ms_monet_model_path
            )

        self.intensity_range_covid = (-1000, 400, 0.0, 1.0, True)
        self.pix_dim = (2.0, 2.0, 1.0)
        self.lamda = 2.5
        self.sigma = 0.15
        self.epochs = 200
        self.tau = 0.3

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Segmentation - Adaptive Online Likelihood Model",
            description="Active Learning solution to label Online Likelihood Model over 3D CT Images",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        models = {
            "COVID_3D_UNet_Segmentation": InferCovid(
                [self.pretrained_model_covid, self.final_model_covid],
                self.network_covid,
            ),
        }

        interactive_models = {}
        interactive_models["MONet"] = OnlineAdaptiveMONetBasedGraphCut(
            intensity_range=self.intensity_range_covid,
            pix_dim=self.pix_dim,
            model="MSFEAT",
            loss="CE",
            train_feat=True,
            lamda=self.lamda,
            sigma=self.sigma,
            tau=self.tau,
            model_path=self.current_ms_monet_model_path,
            lr_scheduler="COSINE",
            epochs=self.epochs,
        )
        interactive_models["MONet-NoMS"] = OnlineAdaptiveMONetBasedGraphCut(
            intensity_range=self.intensity_range_covid,
            pix_dim=self.pix_dim,
            model="FEAT",
            loss="CE",
            train_feat=True,
            lamda=self.lamda,
            sigma=self.sigma,
            tau=self.tau,
            model_path=self.current_ss_econet_model_path,
            lr_scheduler="COSINE",
            epochs=self.epochs,
        )
        interactive_models["ECONet"] = OnlineECONetBasedGraphCut(
            intensity_range=self.intensity_range_covid,
            pix_dim=self.pix_dim,
            model="FEAT",
            loss="CE",
            train_feat=True,
            lamda=self.lamda,
            sigma=self.sigma,
            model_path=self.current_ss_econet_model_path,
            lr_scheduler="COSINE",
            epochs=self.epochs,
        )
        interactive_models["InteractiveGraphcut"] = InteractiveGraphCut(
            intensity_range=self.intensity_range_covid,
            pix_dim=self.pix_dim,
            lamda=self.lamda,
            sigma=self.sigma,
        )

        models.update(interactive_models)

        return models

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {}
        strategies["random"] = Random()
        return strategies

    def infer(self, request, datastore=None):
        image = request.get("image")

        label = request.get("label")

        # add saved logits into request
        if self._infers[request.get("model")].type == InferType.SCRIBBLES:
            saved_labels = self.datastore().get_labels_by_image_id(image)
            for tag, label in saved_labels.items():
                if tag == "logits":
                    request["logits"] = self.datastore().get_label_uri(label, tag)
            logger.info(f"Updated request: {request}")

        result = super().infer(request)
        result_params = result.get("params")

        # save logits
        logits = result_params.get("logits")
        if logits and self._infers[request.get("model")].type == InferType.SEGMENTATION:
            self.datastore().save_label(image, logits, "logits", {})
            os.unlink(logits)

        result_params.pop("logits", None)
        logger.info(f"Final Result: {result}")
        return result

    def next_sample(self, request):
        # Reset MONet/MONet-NoMS/ECONet weights to pretrained weights on next sample
        if os.path.exists(self.current_ms_monet_model_path):
            logging.info(
                "Resetting multi-scale online model for previous sample to pretrained model: {}".format(
                    self.current_ms_monet_model_path
                )
            )
            os.unlink(self.current_ms_monet_model_path)
            shutil.copy(
                self.multiscale_monet_pretrained_path, self.current_ms_monet_model_path
            )

        if os.path.exists(self.current_ss_econet_model_path):
            logging.info(
                "Resetting single-scale online model for previous sample to pretrained model: {}".format(
                    self.current_ss_econet_model_path
                )
            )
            os.unlink(self.current_ss_econet_model_path)
            shutil.copy(
                self.singlescale_econet_pretrained_path,
                self.current_ss_econet_model_path,
            )

        return super().next_sample(request)
