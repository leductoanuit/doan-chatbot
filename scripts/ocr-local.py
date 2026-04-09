"""Local OCR: Warp Affine deskew + Tesseract for all image folders in data/raw.

Usage: python3 data/raw/ocr-local.py
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

RAW_DIR = Path(__file__).parent
OUTPUT_DIR = RAW_DIR.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def deskew_warp_affine(img: np.ndarray) -> np.ndarray:
    """Deskew image using Warp Affine based on detected text angle."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Binarize + invert so text is white on black (for minAreaRect)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all white pixel coordinates (text pixels)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return img  # not enough text to determine angle

    # minAreaRect returns angle of the minimum bounding rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle: minAreaRect returns [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip if angle is negligible
    if abs(angle) < 0.3:
        return img

    # Warp Affine rotation around center
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess(img: np.ndarray) -> np.ndarray:
    """Deskew + enhance for better OCR accuracy."""
    # Step 1: Warp Affine deskew
    img = deskew_warp_affine(img)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Step 3: Adaptive threshold for clean binary text
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return binary


def ocr_image(img_path: Path) -> str:
    """Read image, deskew, OCR with Tesseract (Vietnamese + English)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return ""

    processed = preprocess(img)

    text = pytesseract.image_to_string(
        processed,
        lang="vie+eng",
        config="--psm 6 --oem 3",
    )
    return clean_ocr_text(text)


def clean_ocr_text(text: str) -> str:
    """Remove OCR noise: special chars, excessive whitespace, garbage tokens."""
    # Keep Vietnamese chars, alphanumeric, common punctuation
    # Vietnamese diacritics: àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ
    text = re.sub(
        r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ.,;:!?()\"'\-/§°%&@#\d]",
        "",
        text,
    )
    # Collapse multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just 1-2 garbage chars
    lines = [l for l in text.split("\n") if len(l.strip()) > 2]
    return "\n".join(lines).strip()


def main():
    folders = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(folders)} folders")
    print("=" * 60)

    all_chunks = []

    for fi, folder in enumerate(folders, 1):
        images = sorted(folder.glob("*.png"))
        if not images:
            continue

        print(f"\n[{fi}/{len(folders)}] {folder.name} ({len(images)} pages)")
        doc_chunks = []

        for pi, img_path in enumerate(images, 1):
            text = ocr_image(img_path)
            chars = len(text)
            print(f"  Page {pi}/{len(images)}: {chars} chars")

            if text:
                doc_chunks.append({
                    "content": text,
                    "page": pi,
                    "source": folder.name + ".pdf",
                    "method": "tesseract_warp_affine",
                })

        all_chunks.extend(doc_chunks)
        print(f"  -> {len(doc_chunks)} pages with text")

    # Save output
    output_path = OUTPUT_DIR / "all_documents_ocr.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Done. {len(all_chunks)} chunks from {len(folders)} folders")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
