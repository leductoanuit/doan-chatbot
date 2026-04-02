# Phase 2: DeepSeek OCR Google Colab Notebook

## Context
- Current: `pdf_extractor.py` calls `DEEPSEEK_OCR_URL` (default `http://localhost:5000/ocr`)
- Need: Colab notebook that runs DeepSeek OCR model on free GPU, exposes via ngrok

## Overview
- **Priority**: High
- **Status**: Pending
- **Description**: Create Colab notebook running DeepSeek OCR service with GPU, accessible via ngrok tunnel

## Requirements
### Functional
- Load DeepSeek VL2 or OCR model on Colab GPU
- Expose `/ocr` endpoint accepting `{"image": "<base64>"}`, returning `{"text": "..."}`
- Compatible with existing `extract_text_ocr()` client code
- Support batch processing endpoint for multiple pages

### Non-functional
- Response time < 10s per page on Colab T4 GPU
- Auto-restart on Colab disconnect
- Print ngrok URL for easy copy to `.env`

## Related Code Files
- **Create**: `notebooks/deepseek-ocr-service.ipynb` — Colab notebook
- **Modify**: `.env.example` — add `DEEPSEEK_OCR_URL` documentation

## Implementation Steps

1. Create `notebooks/deepseek-ocr-service.ipynb` with cells:
   
   **Cell 1: Install dependencies**
   ```python
   !pip install flask pyngrok transformers torch Pillow
   ```

   **Cell 2: Setup ngrok**
   ```python
   from pyngrok import ngrok
   # User sets their ngrok authtoken
   ngrok.set_auth_token("YOUR_NGROK_TOKEN")
   ```

   **Cell 3: Load DeepSeek model**
   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "deepseek-ai/deepseek-vl2-tiny"  # or appropriate OCR model
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   model = AutoModelForCausalLM.from_pretrained(
       model_name, trust_remote_code=True,
       torch_dtype=torch.float16
   ).cuda()
   ```

   **Cell 4: Flask OCR server**
   ```python
   import base64, io
   from flask import Flask, request, jsonify
   from PIL import Image

   app = Flask(__name__)

   @app.route('/ocr', methods=['POST'])
   def ocr():
       data = request.json
       img_b64 = data['image']
       img_bytes = base64.b64decode(img_b64)
       image = Image.open(io.BytesIO(img_bytes))

       # OCR inference
       prompt = "OCR this document image. Extract all Vietnamese text exactly as written."
       # Model-specific inference code here
       text = run_ocr_inference(model, tokenizer, image, prompt)

       return jsonify({"text": text})

   @app.route('/health', methods=['GET'])
   def health():
       return jsonify({"status": "ok", "gpu": torch.cuda.is_available()})
   ```

   **Cell 5: Start server with ngrok**
   ```python
   public_url = ngrok.connect(5000)
   print(f"DEEPSEEK_OCR_URL={public_url}/ocr")
   app.run(port=5000)
   ```

2. Add `.env.example` entry:
   ```
   DEEPSEEK_OCR_URL=https://xxxx.ngrok.io/ocr
   ```

## Todo
- [ ] Create `notebooks/deepseek-ocr-service.ipynb`
- [ ] Test with sample images on Colab T4
- [ ] Verify compatibility with `pdf_extractor.py` client
- [ ] Add health check endpoint
- [ ] Document ngrok setup in notebook markdown cells

## Success Criteria
- Notebook runs on Colab free tier (T4 GPU)
- `/ocr` endpoint returns Vietnamese text from base64 image
- Compatible with existing `extract_text_ocr()` without code changes (same API contract)

## Risk Assessment
- **Colab session timeout**: 90min idle → add keep-alive cell or document reconnection
- **ngrok URL changes on restart**: User must update `.env` each time → document clearly
- **Model size**: DeepSeek VL2 may exceed free tier RAM → use `tiny` variant or quantized
