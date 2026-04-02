# Phase 1: PDF Image Preprocessing (Warp Affine)

## Context
- Current: `src/scraper/pdf_extractor.py` renders PDF pages at 2x zoom, sends raw image to OCR
- Problem: Scanned PDFs often skewed/rotated, reducing OCR accuracy
- Solution: Add OpenCV deskew pipeline before OCR

## Overview
- **Priority**: High
- **Status**: Pending
- **Description**: Add image preprocessing (deskew, denoise, binarize) using OpenCV Warp Affine before sending to DeepSeek OCR

## Key Insights
- Hough Line Transform or `cv2.minAreaRect` on contours detect skew angle
- `cv2.warpAffine` with rotation matrix corrects skew
- Preprocessing order: grayscale → denoise → binarize → deskew
- Vietnamese documents often have stamps/seals that can confuse angle detection — use text-region-only approach

## Requirements
### Functional
- Detect skew angle of scanned PDF page images
- Apply Warp Affine transformation to correct alignment
- Apply denoising and adaptive binarization for OCR quality
- Integrate into existing `extract_text_ocr()` flow

### Non-functional
- Processing time < 2s per page
- Must not degrade already-straight images

## Architecture
```
PDF Page → PyMuPDF render (2x) → grayscale → GaussianBlur denoise
  → Otsu binarize → detect skew angle → cv2.warpAffine deskew
  → clean image → base64 encode → DeepSeek OCR API
```

## Related Code Files
- **Modify**: `src/scraper/pdf_extractor.py` — add preprocessing before OCR call
- **Create**: `src/scraper/image-preprocessor.py` — deskew + denoise + binarize module

## Implementation Steps

1. Add `opencv-python-headless` and `numpy` to `requirements.txt`
2. Create `src/scraper/image-preprocessor.py`:
   ```python
   import cv2
   import numpy as np

   def deskew_image(img_bytes: bytes) -> bytes:
       """Deskew a page image using Warp Affine."""
       nparr = np.frombuffer(img_bytes, np.uint8)
       img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

       # Grayscale + blur + binarize
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       gray = cv2.GaussianBlur(gray, (5, 5), 0)
       _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

       # Find contours, get min area rect for skew angle
       coords = np.column_stack(np.where(binary > 0))
       angle = cv2.minAreaRect(coords)[-1]

       # Normalize angle
       if angle < -45:
           angle = -(90 + angle)
       else:
           angle = -angle

       # Skip if angle negligible
       if abs(angle) < 0.5:
           return img_bytes

       # Warp Affine rotation
       (h, w) = img.shape[:2]
       center = (w // 2, h // 2)
       M = cv2.getRotationMatrix2D(center, angle, 1.0)
       rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

       # Encode back to PNG bytes
       _, buf = cv2.imencode('.png', rotated)
       return buf.tobytes()

   def preprocess_for_ocr(img_bytes: bytes) -> bytes:
       """Full preprocessing: deskew → denoise → enhance contrast."""
       deskewed = deskew_image(img_bytes)
       nparr = np.frombuffer(deskewed, np.uint8)
       img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

       # Adaptive threshold for clean text
       denoised = cv2.fastNlMeansDenoising(img, h=10)
       enhanced = cv2.adaptiveThreshold(
           denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
           cv2.THRESH_BINARY, 11, 2
       )

       _, buf = cv2.imencode('.png', enhanced)
       return buf.tobytes()
   ```

3. Update `src/scraper/pdf_extractor.py` `extract_text_ocr()`:
   - After `pix.tobytes("png")`, call `preprocess_for_ocr(img_bytes)`
   - Then base64 encode the preprocessed image

## Todo
- [ ] Add opencv-python-headless to requirements.txt
- [ ] Create `src/scraper/image-preprocessor.py`
- [ ] Update `extract_text_ocr()` to use preprocessor
- [ ] Test with sample skewed PDFs

## Success Criteria
- Skewed PDFs produce straight images (visual check)
- OCR text output improves vs raw images on sample set
- No regression on already-straight PDFs

## Risk Assessment
- **Stamps/seals**: May confuse angle detection → mitigate with angle threshold (skip if < 0.5 deg)
- **Heavy images**: Large pages slow to process → 2x zoom already reasonable, preprocessing adds ~1s
