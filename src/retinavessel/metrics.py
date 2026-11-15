"""Retina-only pixel, centerline, connectivity and thin-vessel metrics."""
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    label,
)
from skimage.morphology import skeletonize


def approx_fov(image, threshold=10):
    """Image-derived approximate FOV. Audit threshold; replace with published FOV masks when present."""
    rgb = np.asarray(image.convert('RGB'))
    initial = rgb.max(axis=2) > threshold
    filled = binary_fill_holes(binary_closing(initial, structure=np.ones((7,7), dtype=bool)))
    labels, n = label(filled)
    if n == 0:
        raise ValueError('Cannot determine retinal field of view')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax()


def _component_count(skeleton, minimum_pixels=5):
    labels, count = label(skeleton, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return 0
    sizes = np.bincount(labels.ravel())
    return int(np.sum(sizes[1:] >= minimum_pixels))


def compute_metrics(
    prob,
    ground_truth,
    fov,
    threshold=0.5,
    include_topology=False,
    thin_max_radius=2.0,
    thin_tolerance=1,
):
    truth = np.asarray(ground_truth, dtype=bool)
    fov = np.asarray(fov, dtype=bool)
    pred = np.asarray(prob) >= threshold
    if not fov.any():
        raise ValueError('Empty FOV')
    tp = int(np.sum(pred & truth & fov))
    fp = int(np.sum(pred & ~truth & fov))
    fn = int(np.sum(~pred & truth & fov))
    tn = int(np.sum(~pred & ~truth & fov))
    def ratio(a,b): return float(a/b) if b else 0.0
    out = {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
           'dice':ratio(2*tp,2*tp+fp+fn),
           'iou':ratio(tp,tp+fp+fn),
           'sensitivity':ratio(tp,tp+fn),
           'specificity':ratio(tn,tn+fp),
           'precision':ratio(tp,tp+fp)}
    if include_topology:
        pskel = skeletonize(pred & fov)
        tskel = skeletonize(truth & fov)
        tprec = ratio((pskel & truth).sum(), pskel.sum())
        tsens = ratio((tskel & pred).sum(), tskel.sum())
        out['cldice'] = ratio(2*tprec*tsens, tprec+tsens)
        radius = distance_transform_edt(truth & fov)
        thin_centerline = tskel & (radius <= float(thin_max_radius))
        if thin_tolerance < 0:
            raise ValueError('thin_tolerance must be non-negative')
        detected = pred & fov
        if thin_tolerance:
            detected = binary_dilation(
                detected,
                structure=np.ones((3, 3), dtype=bool),
                iterations=int(thin_tolerance),
            )
        thin_pixels = int(thin_centerline.sum())
        out['thin_centerline_pixels'] = thin_pixels
        out['thin_centerline_recall'] = ratio(
            (thin_centerline & detected).sum(), thin_pixels
        )
        predicted_components = _component_count(pskel)
        truth_components = _component_count(tskel)
        out['predicted_skeleton_components'] = predicted_components
        out['truth_skeleton_components'] = truth_components
        out['skeleton_component_ratio'] = ratio(
            predicted_components, truth_components
        )
    return out
