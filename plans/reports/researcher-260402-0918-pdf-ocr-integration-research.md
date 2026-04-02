# Research Report: PDF Deskewing, Qdrant VectorDB, and Colab Integration

**Date:** 2026-04-02 | **Status:** Complete

---

## Topic 1: OpenCV Warp Affine for PDF Deskewing

### Skew Detection
Two approaches dominate:
1. **MinAreaRect (simpler):** Compute minimum rotated rectangle on foreground pixels
   ```python
   coords = np.column_stack(np.where(thresh > 0))
   angle = cv2.minAreaRect(coords)[-1]
   if angle < -45:
       angle = -(90 + angle)
   ```

2. **Hough Transform (advanced):** Uses probabilistic Hough line detection for fragmented lines; more robust on damaged PDFs

### Preprocessing Pipeline
```python
# Grayscale + invert + binarization (Otsu's)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
```

### Deskewing (warpAffine)
```python
(h, w) = img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h), 
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
```

**Best practice:** Use `INTER_CUBIC` + `BORDER_REPLICATE` for quality PDF output; minAreaRect sufficient for most scans.

---

## Topic 2: Qdrant Vector Database

### Setup & Collection Creation
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(":memory:")  # or URL/API key
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
```

### Upserting with Metadata
```python
from qdrant_client.http.models import PointStruct

client.upsert(
    collection_name="docs",
    points=[
        PointStruct(id=1, vector=[...768 dims...], 
                   payload={"source": "page1.pdf", "text": "..."})
    ]
)
```

### Filtered Search
```python
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="docs",
    query_vector=[...768 dims...],
    query_filter=Filter(must=[
        FieldCondition(key="source", match=MatchValue(value="page1.pdf"))
    ]),
    limit=5
)
```

### Deployment
- **Docker:** `docker run -p 6333:6333 qdrant/qdrant` (local, no auth)
- **Qdrant Cloud:** Managed service with API key (free tier: 1GB)

**vs MongoDB Atlas:** Qdrant faster for similarity search; MongoDB better for hybrid text+vector; Qdrant simpler setup.

---

## Topic 3: Google Colab FastAPI + ngrok + DeepSeek OCR

### Minimal Setup
```python
!pip install fastapi uvicorn nest-asyncio pyngrok

import nest_asyncio
from fastapi import FastAPI
from pyngrok import ngrok

app = FastAPI()

@app.post('/ocr')
async def ocr_endpoint(file: UploadFile):
    # DeepSeek OCR inference here
    return {"text": "..."}

ngrok_tunnel = ngrok.connect(8000)
print('URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
```

### DeepSeek-OCR on T4 GPU
- Requires 4-bit quantization to fit T4 VRAM (~8GB free)
- Load via HuggingFace with `load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16`
- Inference: ~2-5s/page on T4

### Key Requirements
- **ngrok authtoken:** Required (free account at ngrok.com)
- **CORS:** Add middleware for frontend access
- **Timeout:** Increase FastAPI worker timeout for OCR jobs (30-60s)

**Limitation:** Colab notebooks auto-disconnect after inactivity; not suitable for production. Use for development/demos only.

---

## Integration Pattern (Recommended)

```
PDF Upload → Deskew (warpAffine) → DeepSeek OCR (Colab GPU) 
→ Embed (sentence-transformers 768-dim) → Upsert to Qdrant 
→ Search & Filter via FastAPI
```

---

## Unresolved Questions

1. **Hough vs minAreaRect trade-off:** Performance benchmark on heavily skewed PDFs needed
2. **Qdrant persistence:** Best practice for local Docker storage (volumes vs disk)?
3. **Colab GPU reliability:** How to handle ngrok disconnections gracefully in production?
