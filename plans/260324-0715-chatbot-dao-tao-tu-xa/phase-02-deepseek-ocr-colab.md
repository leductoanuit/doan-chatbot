# Phase 2: DeepSeek OCR on Google Colab

## Priority: High | Status: ✅ | Effort: Medium

## Overview
Replace PaddleOCR with DeepSeek OCR model deployed as an API on Google Colab. The Colab notebook runs the DeepSeek model and exposes an HTTP endpoint (via ngrok or Colab's built-in tunneling). The local `pdf-extractor.py` calls this API instead of running OCR locally.

## Files to Modify
- `src/scraper/pdf-extractor.py` — replace `extract_text_ocr()` with DeepSeek API call
- `notebooks/deepseek-ocr-colab.ipynb` — **new file**, Colab notebook for DeepSeek OCR server

## Files to Create
- `notebooks/deepseek-ocr-colab.ipynb` — Colab notebook

## Implementation Steps

### 1. Create Colab Notebook (`notebooks/deepseek-ocr-colab.ipynb`)
```python
# Cell 1: Install dependencies
!pip install transformers torch flask pyngrok Pillow PyMuPDF

# Cell 2: Load DeepSeek OCR model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "deepseek-ai/deepseek-vl2-tiny"  # or appropriate OCR model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.float16
).cuda()

# Cell 3: Flask API server
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io

app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.json
    image_b64 = data["image"]
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    # Run OCR inference
    # ... model-specific inference code ...
    return jsonify({"text": extracted_text})

# Cell 4: Expose via ngrok
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(f"OCR API URL: {public_url}")
app.run(port=5000)
```

### 2. Update `pdf-extractor.py`
- Replace `extract_text_ocr()` to call DeepSeek API instead of PaddleOCR
- Send each page as base64 image to Colab endpoint
- Read OCR API URL from env var `DEEPSEEK_OCR_URL`

```python
def extract_text_ocr(pdf_path: str) -> list[dict]:
    """Send PDF pages as images to DeepSeek OCR API on Colab."""
    import fitz
    import base64
    import requests as req
    from PIL import Image

    ocr_url = os.getenv("DEEPSEEK_OCR_URL", "http://localhost:5000/ocr")
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        resp = req.post(ocr_url, json={"image": img_b64}, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()

        if text:
            chunks.append({
                "content": text,
                "page": page_num + 1,
                "source": os.path.basename(pdf_path),
                "method": "deepseek_ocr",
            })

    doc.close()
    return chunks
```

### 3. Update `.env.example`
- Add `DEEPSEEK_OCR_URL=` entry

## Todo
- [x] Create Colab notebook with DeepSeek model loading + Flask API
- [x] Update pdf-extractor.py extract_text_ocr() to call API
- [x] Add DEEPSEEK_OCR_URL to .env.example
- [x] Remove PaddleOCR imports/dependencies from pdf-extractor.py
- [x] Update requirements.txt (remove paddleocr/paddlepaddle)
- [x] Test with a sample PDF

## Success Criteria
- Colab notebook starts, loads DeepSeek model, exposes API via ngrok
- pdf-extractor.py sends page images to API and receives Vietnamese text back
- Image-based PDFs from daa.uit.edu.vn are correctly OCR'd

## Risk
- Colab session timeout (max ~12h) — document this limitation
- ngrok URL changes each session — must update env var
- DeepSeek model may need specific prompting for Vietnamese OCR — test and adjust
- Large PDFs may timeout — add retry logic with backoff
