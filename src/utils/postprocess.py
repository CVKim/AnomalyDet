"""Convert anomaly heatmaps to binary masks and LabelMe-compatible JSON."""
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def foreground_mask(image_rgb: np.ndarray,
                    bg_threshold: int = 30,
                    morph_kernel: int = 21,
                    keep_largest: bool = True) -> np.ndarray:
    """Cheap foreground saliency for parts photographed on a dark
    background (MVTec hazelnut, screw, etc.). Converts to grayscale,
    Otsu-threshold-ish split + morphological close, optionally keeps
    only the largest connected component. Returns a uint8 {0, 255}
    mask the same H x W as the input.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY)
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (morph_kernel, morph_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if keep_largest:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = np.where(labels == largest, 255, 0).astype(np.uint8)
    return binary


def threshold_heatmap(heatmap: np.ndarray,
                      threshold: Optional[float] = None,
                      percentile: Optional[float] = None) -> np.ndarray:
    if percentile is not None:
        threshold = float(np.percentile(heatmap, percentile))
    if threshold is None:
        raise ValueError('Provide either threshold or percentile.')
    return ((heatmap >= threshold).astype(np.uint8)) * 255


def adaptive_pixel_threshold(heatmap: np.ndarray,
                             image_score: float,
                             train_image_max: Optional[float],
                             train_pixel_max: Optional[float],
                             image_gate_factor: float = 1.3,
                             severity_fraction: float = 0.5,
                             pixel_floor_factor: float = 1.1) -> float:
    """Per-image threshold combining three signals:
      1. Image-level gate: if image_score < train_image_max * image_gate_factor,
         the image is treated as normal and an empty mask is returned (+inf).
      2. Otsu's bimodal split on this image's heatmap distribution.
      3. severity_fraction * image_score, which tracks the image's own peak
         so that low-severity defects still produce tight, not exploded, masks.
      4. pixel_floor_factor * train_pixel_max as a hard floor so we never
         drop below known noise.

    Final threshold = max(Otsu, severity, pixel_floor). The intent is recall-
    first detection (gate is loose) plus a tight, useful mask once detected.
    """
    if train_image_max is not None and image_score < train_image_max * image_gate_factor:
        return float('inf')

    rng = float(heatmap.max() - heatmap.min())
    if rng < 1e-6:
        return float('inf')
    h8 = ((heatmap - heatmap.min()) / rng * 255.0).astype(np.uint8)
    otsu_t_u8, _ = cv2.threshold(h8, 0, 255, cv2.THRESH_OTSU)
    t_otsu = float(otsu_t_u8) / 255.0 * rng + float(heatmap.min())
    t_severity = float(image_score) * float(severity_fraction)
    t_floor = (float(train_pixel_max) * float(pixel_floor_factor)
               if train_pixel_max is not None else 0.0)
    return max(t_otsu, t_severity, t_floor)


def clean_mask(mask: np.ndarray, kernel_size: int = 5,
               min_area: int = 50,
               merge_kernel: int = 0) -> np.ndarray:
    """Morphological cleanup + min-area filter.

    merge_kernel > 0 dilates by that radius before reading components,
    then erodes back so fragmented blobs that sit within ~merge_kernel
    pixels of each other are read as a single defect region. This
    matters on textured surfaces (hazelnut shell) where the raw
    threshold output is locally fragmented.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if merge_kernel > 1:
        mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_kernel, merge_kernel))
        out = cv2.dilate(out, mk)
        out = cv2.erode(out, mk)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    cleaned = np.zeros_like(out)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def mask_to_labelme_json(mask: np.ndarray,
                         image_path: str,
                         image_shape,
                         image_score: Optional[float] = None,
                         defect_label: str = 'defect',
                         poly_eps_ratio: float = 0.005) -> dict:
    """LabelMe-style JSON. Each connected region becomes a polygon shape."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for c in contours:
        if len(c) < 3:
            continue
        eps = poly_eps_ratio * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).squeeze(1)
        if approx.ndim != 2 or approx.shape[0] < 3:
            continue
        x, y, w, h = cv2.boundingRect(c)
        shapes.append({
            'label': defect_label,
            'points': approx.astype(float).tolist(),
            'shape_type': 'polygon',
            'bbox_xywh': [int(x), int(y), int(w), int(h)],
            'group_id': None,
            'flags': {},
        })
    return {
        'version': '5.3.1',
        'flags': {},
        'shapes': shapes,
        'imagePath': str(Path(image_path).name),
        'imageData': None,
        'imageHeight': int(image_shape[0]),
        'imageWidth': int(image_shape[1]),
        'imageScore': float(image_score) if image_score is not None else None,
    }


def save_outputs(out_dir: str, image_name: str,
                 heatmap_norm: np.ndarray,
                 mask: np.ndarray,
                 json_data: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem

    hm_uint8 = (np.clip(heatmap_norm, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(str(out / f'{stem}_heatmap.png'), hm_uint8)
    cv2.imwrite(str(out / f'{stem}_mask.png'), mask)
    json_path = out / f'{stem}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    return {
        'heatmap_png': str(out / f'{stem}_heatmap.png'),
        'mask_png': str(out / f'{stem}_mask.png'),
        'json': str(json_path),
    }
