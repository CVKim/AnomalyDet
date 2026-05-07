import cv2
import numpy as np


def normalize_map(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


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
