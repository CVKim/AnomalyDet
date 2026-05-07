import cv2
import numpy as np


def normalize_map(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def normalize_map_calibrated(arr: np.ndarray,
                             train_pixel_max: float,
                             anomaly_scale: float = 2.5,
                             normal_band: float = 0.3) -> np.ndarray:
    """Calibrated [0,1] colormap that anchors against the training-set
    pixel ceiling so good and defective images render on the same scale.

    Score <= train_pixel_max:           linearly mapped to [0, normal_band]
                                        (blue/green region — definitely normal)
    train_pixel_max < score <= scale*pmax:  linearly mapped to (normal_band, 1]
                                        (yellow -> red — gradient anomaly)
    Score > scale*pmax:                 saturates at 1 (deep red)

    With this, a clean image stays mostly cool-coloured and anomalies are
    the only red area, instead of every image getting a per-image min-max
    stretch that paints natural texture red.
    """
    if train_pixel_max is None or train_pixel_max <= 0:
        return normalize_map(arr)
    pmax = float(train_pixel_max)
    upper = pmax * float(anomaly_scale)
    out = np.zeros_like(arr, dtype=np.float32)
    below = arr <= pmax
    out[below] = (arr[below] / max(pmax, 1e-6)) * float(normal_band)
    above = ~below
    span = max(upper - pmax, 1e-6)
    out[above] = float(normal_band) + np.clip(
        (arr[above] - pmax) / span, 0.0, 1.0
    ) * (1.0 - float(normal_band))
    return out


def overlay_heatmap(image_rgb: np.ndarray, heatmap_norm: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    hm = (np.clip(heatmap_norm, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, color, alpha, 0)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color=(255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[mask > 0] = np.array(color, dtype=overlay.dtype)
    return cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)


def draw_contours(image_rgb: np.ndarray, mask: np.ndarray,
                  color=(255, 0, 0), thickness: int = 2) -> np.ndarray:
    out = image_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out
