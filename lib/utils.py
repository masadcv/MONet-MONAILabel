import logging
from functools import partial

import FastGeodis
import GeodisTK
import numpy as np
import numpymaxflow
import torch


def get_eps(data):
    return np.finfo(data.dtype).eps


def maxflow(image, prob, lamda=5, sigma=0.1):
    # lamda: weight of smoothing term
    # sigma: std of intensity values
    return numpymaxflow.maxflow(image, prob, lamda, sigma)


###################################################
###    Geodesic Distance Transform Functions
###################################################


def fastgeodis_generalised_geodesic_distance_3d(I, S, spacing, v, lamda=1.0, iter=4):
    return FastGeodis.generalised_geodesic3d(I, S, spacing, v, lamda, iter)


def fastgeodis_generalised_geodesic_distance_2d(I, S, v, lamda=1.0, iter=2):
    return FastGeodis.generalised_geodesic3d(I, S, v, lamda, iter)


def geodesic_distance_3d(I, S, spacing, lamda=1.0, iter=4):
    # lamda=0 : euclidean dist
    # lamda=1 : geodesic dist
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamda, iter)


def geodesic_distance_2d(I, S, lamda=1.0, iter=2):
    # lamda=0 : euclidean dist
    # lamda=1 : geodesic dist
    return GeodisTK.geodesic2d_raster_scan(I, S, lamda, iter)


def compute_scribbles_geodesic(
    image,
    scribbles,
    scribbles_fg_label,
    scribbles_bg_label,
    spacing,
    run_3d,
    use_fastgeodis,
    device,
):
    # get Geodesic Distances for foreground (D_fg) and background (D_bg) scribbles
    S_fg = (scribbles == scribbles_fg_label).astype(np.uint8)
    S_bg = (scribbles == scribbles_bg_label).astype(np.uint8)

    if use_fastgeodis:
        func3d = partial(fastgeodis_generalised_geodesic_distance_3d, v=1e10)
        func2d = partial(fastgeodis_generalised_geodesic_distance_2d, v=1e10)
        # convert data to torch tensor and load to device
        image = (
            torch.from_numpy(image)
            .unsqueeze(0)
            .unsqueeze(0)
            .type(torch.float32)
            .to(device)
        )
        S_fg = 1 - torch.from_numpy(S_fg).unsqueeze(0).unsqueeze(0).type(
            torch.float32
        ).to(device)
        S_bg = 1 - torch.from_numpy(S_bg).unsqueeze(0).unsqueeze(0).type(
            torch.float32
        ).to(device)
        logging.info("Geodesic using FastGeodis")

    else:
        func3d = geodesic_distance_3d
        func2d = geodesic_distance_2d
        logging.info("Geodesic using GeodisTK")

    # run geodesic distance transform
    if run_3d:
        D_fg = func3d(image, S_fg, spacing, lamda=1.0)
        D_bg = func3d(image, S_bg, spacing, lamda=1.0)
    else:
        D_fg = func2d(image, S_fg, lamda=1.0)
        D_bg = func2d(image, S_bg, lamda=1.0)

    if use_fastgeodis:
        D_fg = np.squeeze(D_fg.detach().cpu().numpy())
        D_bg = np.squeeze(D_bg.detach().cpu().numpy())

    return D_fg, D_bg


###################################################

###################################################
###    Adaptive MONet Exponential Geodesic Function
###################################################


def make_egd_adaptive_weights(
    image,
    scribbles,
    scribbles_fg_label,
    scribbles_bg_label,
    spacing,
    tau=1.0,
    use_fastgeodis=False,
    device="cuda",
):
    # inputs are expected to be of format [1, X, Y, [Z]]
    # simple check to see if input shape is expected
    if image.shape[0] != 1 or scribbles.shape[0] != image.shape[0]:
        raise ValueError("unexpected input shape received")

    # extract spatial dims
    spatial_dims = image.ndim - 1
    run_3d = spatial_dims == 3

    scribbles = np.squeeze(scribbles)
    image = np.squeeze(image)

    # get Geodesic Distances for foreground (D_fg) and background (D_bg) scribbles
    D_fg, D_bg = compute_scribbles_geodesic(
        image,
        scribbles,
        scribbles_fg_label,
        scribbles_bg_label,
        spacing,
        run_3d,
        use_fastgeodis,
        device,
    )
    # do ExponentialGaussianDistance calculation
    D_g = np.array([D_bg, D_fg])

    # calculate alpha_i for each element to be updated
    alpha_i = np.exp(-D_g / (tau + get_eps(D_g)))

    return alpha_i


###################################################

###################################################
###   Probability Guided Pruning Functions
###################################################


def make_uncertainity_from_logits(data, axis=0):
    c_data = -data * np.log2(data + get_eps(data))
    return np.sum(c_data, axis=axis)


def create_drop_value_mask(shape, dropout=0.5):
    mask = np.random.uniform(size=shape)
    mask = (mask >= dropout).astype(np.float32)

    s_frac = (np.sum(mask == 0) / mask.size) * 100
    # s_frac = np.sum((mask == 0.0) == (
    # mask == 0.0)) / np.sum(mask == 0.0)
    logging.info("Mask dropout frac achieved: {}".format(s_frac))

    return mask


def create_drop_prob_mask(prob, confidence=0.5, inequality_less=False):
    # get threshold for quantile=confidence
    # threshold = np.quantile(prob.flatten(), q=confidence)
    mask = prob >= confidence
    if inequality_less:
        mask = (~mask).astype(np.float32).copy()

    s_frac = (np.sum(mask == 0) / mask.size) * 100
    # s_frac = np.sum((mask == 0.0) == (
    # mask == 0.0)) / np.sum(mask == 0.0)
    logging.info("Mask dropout frac achieved: {}".format(s_frac))
    return mask


###################################################
