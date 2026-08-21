"""
Hand detection & cropping module.
Uses MediaPipe to locate the hand in an image and crop tightly around it,
so the model focuses only on the hand (not the background) for better accuracy.

MediaPipe 1.x uses the new Tasks API (`HandLandmarker`) which requires a
`hand_landmarker.task` model file. If it's missing/unusable, we fall back to
a centered crop.
"""
import os
import numpy as np

MODEL_FILE = "hand_landmarker.task"

_HANDS = None
_HAND_AVAILABLE = False


def _load_landmarker():
    """Lazily build a MediaPipe HandLandmarker using the new Tasks API."""
    global _HANDS, _HAND_AVAILABLE
    if _HANDS is not None or _HAND_AVAILABLE:
        return _HANDS

    if not os.path.exists(MODEL_FILE):
        print(f"[WARN] {MODEL_FILE} not found. Hand crop disabled (center crop).")
        return None

    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
        _HANDS = vision.HandLandmarker.create_from_options(options)
        _HAND_AVAILABLE = True
        return _HANDS
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] MediaPipe hand detector failed to load: {e}")
        _HAND_AVAILABLE = False
        return None


def hand_available():
    """Return True if MediaPipe hand detection is usable."""
    _load_landmarker()
    return _HAND_AVAILABLE


def _to_mp_image(img_np):
    """Convert an RGB numpy array to a MediaPipe Image."""
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)


def _center_crop(img_np, margin=0.15):
    """Fallback: crop a centered square region (roughly the hand area)."""
    h, w = img_np.shape[:2]
    s = min(h, w)
    if s <= 0:
        return img_np
    m = int(s * margin)
    x0, y0 = (w - s) // 2, (h - s) // 2
    x0 = max(0, x0 - m)
    y0 = max(0, y0 - m)
    x1 = min(w, x0 + s + 2 * m)
    y1 = min(h, y0 + s + 2 * m)
    return img_np[y0:y1, x0:x1]


def crop_hand(img_np, pad=0.25):
    """
    Given an RGB numpy image (HxWx3), return a cropped image centered on the hand.
    If no hand is found, returns a centered crop fallback.
    """
    if img_np is None:
        return img_np

    landmarker = _load_landmarker()
    if landmarker is None:
        return _center_crop(img_np)

    h, w = img_np.shape[:2]
    try:
        mp_img = _to_mp_image(img_np)
        result = landmarker.detect(mp_img)
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x0, x1 = min(xs) * w, max(xs) * w
            y0, y1 = min(ys) * h, max(ys) * h

            # Add padding around the hand
            pad_x = (x1 - x0) * pad
            pad_y = (y1 - y0) * pad
            x0 = max(0, int(x0 - pad_x))
            y0 = max(0, int(y0 - pad_y))
            x1 = min(w, int(x1 + pad_x))
            y1 = min(h, int(y1 + pad_y))

            if x1 > x0 and y1 > y0:
                return img_np[y0:y1, x0:x1]
    except Exception as e:  # noqa: BLE001
        print("[WARN] Hand detection failed:", e)

    return _center_crop(img_np)


def crop_hand_pil(pil_img):
    """Crop a hand from a PIL image, returning a PIL image."""
    np_img = np.array(pil_img.convert("RGB"))
    cropped = crop_hand(np_img)
    from PIL import Image
    return Image.fromarray(cropped)
