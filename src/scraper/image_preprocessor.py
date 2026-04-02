"""Image preprocessing for OCR — deskew, denoise, and binarize PNG bytes.

Pipeline: deskew → grayscale → denoise → adaptive threshold binarization.
Designed to improve OCR accuracy on scanned Vietnamese regulation PDFs.
"""

import io
import numpy as np


def deskew_image(img_bytes: bytes) -> bytes:
    """Detect and correct image skew using cv2.minAreaRect on binary contours.

    Skips correction when angle < 0.5 degrees to avoid unnecessary resampling.
    Returns original bytes unchanged if no contours are found or image is empty.
    """
    import cv2

    # Decode PNG bytes to numpy array
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert + threshold to isolate text pixels as white on black
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find all non-zero pixel coordinates for minAreaRect
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        # No text content detected — return original
        return img_bytes

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # minAreaRect returns angle in [-90, 0); map to [-45, 45) range
    if angle < -45:
        angle = 90 + angle

    # Skip trivial rotations to avoid quality loss from unnecessary warp
    if abs(angle) < 0.5:
        return img_bytes

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    deskewed = cv2.warpAffine(
        img,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Encode back to PNG bytes
    success, encoded = cv2.imencode(".png", deskewed)
    if not success:
        return img_bytes
    return encoded.tobytes()


def preprocess_for_ocr(img_bytes: bytes) -> bytes:
    """Full preprocessing pipeline optimized for OCR on scanned documents.

    Steps:
        1. Deskew — correct rotation artifacts from scanning
        2. Grayscale — reduce noise channels
        3. Denoise — fastNlMeansDenoising for salt-and-pepper noise removal
        4. Adaptive threshold binarization — improve contrast for OCR engines

    Returns PNG bytes. Falls back to original bytes on any processing error.
    """
    import cv2

    try:
        # Step 1: Deskew
        img_bytes = deskew_image(img_bytes)

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return img_bytes

        # Step 2: Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3: Denoise — h=10 balances noise removal vs. fine text preservation
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # Step 4: Adaptive threshold binarization — handles uneven page illumination
        binary = cv2.adaptiveThreshold(
            denoised,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )

        success, encoded = cv2.imencode(".png", binary)
        if not success:
            return img_bytes
        return encoded.tobytes()

    except Exception as exc:
        # Log but never crash the OCR pipeline
        print(f"[image-preprocessor] preprocessing failed, using raw image: {exc}")
        return img_bytes
