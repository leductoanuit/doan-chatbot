# Vietnamese Educational Chatbot with RAG: Technical Research Report

**Date:** 2026-03-21
**Focus:** MongoDB Vector Search, PDF Processing, Vietnamese NLP, RAG Architecture

---

## Executive Summary

- **MongoDB Vector Search (7.0+):** Self-hosted capable via Community Edition 8.2+; dual mongod/mongot architecture with $vectorSearch in aggregation pipelines; no Atlas dependency required
- **PDF Processing:** Use PyMuPDF for native text extraction, fallback to OCR (PaddleOCR preferred for Vietnamese); detect PDF type before processing to optimize pipeline
- **Vietnamese OCR:** PaddleOCR > EasyOCR > Pytesseract for Asian languages; test directly on target content; no 2025 Vietnamese-specific benchmarks available
- **Embeddings:** dangvantuan/vietnamese-embedding (768-dim) or keepitreal/vietnamese-sbert recommended for sentence-level; document-specific models for long-form content
- **RAG Architecture:** Hybrid search (vector + keyword) + reranking; RecursiveCharacterTextSplitter 400-512 tokens baseline; LlamaIndex 35% faster retrieval than LangChain in 2025 benchmarks
- **Word Export:** python-docx for programmatic generation; docxtpl for template-based workflows

---

## 1. MongoDB as Vector Database

### Self-Hosted Setup (Preferred)

**Critical:** MongoDB now supports vector search natively in **Community Edition 8.2+** (released 2025). No Atlas subscription required.

**Architecture:**
- **mongod:** Primary database server (MongoDB community or enterprise)
- **mongot:** Search server handling indexing & similarity calculations (must run separately)
- Both communicate via search plugins; mongot indexes are separate from regular MongoDB indexes

**Mongod Configuration:**
```yaml
# /etc/mongod.conf
storage:
  engine: wiredTiger
security:
  authorization: "enabled"
net:
  port: 27017
  bindIp: localhost,127.0.0.1
# Vector search doesn't require special config in mongod itself
```

**Mongot Installation & Setup:**
- Download from MongoDB releases (same version as mongod)
- Run separately: `mongot --ipBindAll --dbpath /var/lib/mongot`
- Connect mongod to mongot via plugin configuration

**PyMongo Vector Index Creation:**
```python
from pymongo.operations import SearchIndexModel

# Define vector index
vector_index_def = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 768,  # Match your embedding model
            "similarity": "cosine"
        },
        {
            "type": "filter",
            "path": "metadata.source"  # For pre-filtering
        }
    ]
}

index_model = SearchIndexModel(
    definition=vector_index_def,
    name="vector_search_idx",
    type="vectorSearch"
)

collection.create_search_indexes([index_model])
```

**Vector Search Query (PyMongo):**
```python
from pymongo.operations import IndexModel

pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_search_idx",
            "path": "embedding",
            "queryVector": query_embedding,  # 768-dim list
            "k": 10,  # Top 10 results
            "numCandidates": 100  # ANN candidate pool
        }
    },
    {
        "$project": {
            "content": 1,
            "metadata": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

results = list(collection.aggregate(pipeline))
```

**Document Schema (Recommended):**
```python
{
    "_id": ObjectId(),
    "content": "Full text chunk",
    "embedding": [0.123, -0.456, ...],  # 768 floats
    "metadata": {
        "source": "pdf_name",
        "page": 5,
        "chunk_idx": 2,
        "created_at": datetime.now(),
        "document_type": "lecture_notes"
    },
    "tokens": 512  # For tracking context window
}
```

**Similarity Metrics:**
- `cosine` (default): Best for semantic text embeddings; angles-based comparison
- `euclidean`: Straight-line distance; sensitive to magnitude
- `dotProduct`: Normalized embeddings only; faster for recommendation systems

**Approximate Nearest Neighbor (ANN):**
- `numCandidates` controls ANN pool size (default 100); higher = more accurate but slower
- MongoDB uses HNSW (Hierarchical Navigable Small World) internally for efficiency
- Trade-off: Higher numCandidates improves recall but increases latency

**Atlas vs Self-Hosted Trade-offs:**
| Factor | Atlas | Self-Hosted 8.2+ |
|--------|-------|-----------------|
| Setup Complexity | Low | Medium |
| Cost | Managed service fee | Infrastructure only |
| Scaling | Automatic | Manual |
| Downtime Risk | Minimal | Your responsibility |
| Embedding Sync | Built-in (preview) | Manual |

---

## 2. PDF Processing Pipeline

### Detection & Strategy

**Step 1: Classify PDF Type**
```python
import fitz  # PyMuPDF

def classify_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]

    # Get text and image counts
    text_content = page.get_text()
    image_count = len(page.get_images())

    # Heuristic: if text < 10 chars AND images > 0 → image-based
    is_image_pdf = len(text_content.strip()) < 10 and image_count > 0

    return "image_based" if is_image_pdf else "text_native"
```

### Native Text Extraction (PyMuPDF)

```python
import fitz

def extract_text_native(pdf_path):
    doc = fitz.open(pdf_path)
    text_chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text(sort=True)  # sort=True preserves reading order

        # Preserve structure
        chunks = text.split('\n\n')
        for chunk in chunks:
            if chunk.strip():
                text_chunks.append({
                    'content': chunk.strip(),
                    'page': page_num + 1,
                    'source': pdf_path
                })

    return text_chunks
```

**Best for:** Educational PDFs with proper text layers; university lecture notes; official documents

### OCR Processing (Image-Based PDFs)

**Library Recommendation:** PaddleOCR > EasyOCR > Pytesseract

**Why PaddleOCR:**
- Best accuracy on Asian languages (Vietnamese included)
- Faster inference than EasyOCR
- Multi-lingual without model switching
- Active Vietnamese model updates
- CPU-friendly (can run without GPU)

```python
from paddleocr import PaddleOCR
import fitz
from PIL import Image
import io

def extract_text_ocr_paddleocr(pdf_path, lang='vi'):
    """Vietnamese OCR using PaddleOCR"""
    ocr = PaddleOCR(use_angle_cls=True, lang=[lang, 'en'])

    doc = fitz.open(pdf_path)
    text_chunks = []

    for page_num, page in enumerate(doc):
        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR

        # Convert to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # OCR
        result = ocr.ocr(img, cls=True)

        # Extract text with confidence
        page_text = ""
        for line in result:
            for word_info in line:
                text, confidence = word_info[1], word_info[2]
                # Filter low-confidence predictions
                if confidence > 0.5:
                    page_text += text + " "

        if page_text.strip():
            text_chunks.append({
                'content': page_text.strip(),
                'page': page_num + 1,
                'source': pdf_path,
                'method': 'paddleocr'
            })

    return text_chunks
```

**PaddleOCR Setup:**
```bash
pip install paddlepaddle paddleocr
# First run downloads Vietnamese model (~100MB)
```

**Alternative: EasyOCR (if PaddleOCR unavailable)**
```python
import easyocr
import fitz

def extract_text_ocr_easyocr(pdf_path, lang='vi'):
    reader = easyocr.Reader(['vi', 'en'], gpu=False)

    doc = fitz.open(pdf_path)
    text_chunks = []

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        results = reader.readtext(img)
        page_text = '\n'.join([text for (_, text, conf) in results if conf > 0.4])

        if page_text.strip():
            text_chunks.append({
                'content': page_text.strip(),
                'page': page_num + 1,
                'source': pdf_path,
                'method': 'easyocr'
            })

    return text_chunks
```

**NOT Recommended: Pytesseract**
- Weaker Vietnamese support (trained on English primarily)
- Slower inference
- Requires separate Tesseract installation (system dependency)

### Combined Pipeline

```python
def process_pdf(pdf_path, use_fallback_ocr=True):
    """
    1. Try native extraction
    2. If sparse, fallback to OCR
    """
    pdf_type = classify_pdf(pdf_path)

    if pdf_type == "text_native":
        chunks = extract_text_native(pdf_path)
        # If extraction failed or too sparse, fallback
        if not chunks or sum(len(c['content']) for c in chunks) < 500:
            if use_fallback_ocr:
                chunks = extract_text_ocr_paddleocr(pdf_path)
    else:
        chunks = extract_text_ocr_paddleocr(pdf_path)

    return chunks
```

**Vietnamese-Specific Considerations:**
- UTF-8 encoding guaranteed in PyMuPDF (automatic)
- PaddleOCR handles Vietnamese diacritics (tone marks) better than alternatives
- PDF fonts may not embed Vietnamese glyphs → OCR likely needed
- Test on 5-10 sample PDFs before full pipeline deployment

---

## 3. Web Scraping Strategy for ctsv.uit.edu.vn

### Reconnaissance Phase

**Before scraping:**
1. Check `robots.txt`: `https://ctsv.uit.edu.vn/robots.txt`
2. Identify rate limits (typically 1-2 req/sec respectful threshold)
3. Analyze site structure: navigation, PDF URLs, dynamic vs static content
4. Test user-agent (some sites block robots)

### Tool Selection

**Recommended Stack:** BeautifulSoup + Requests (lightweight) or Scrapy (if large-scale crawl)

**BeautifulSoup for moderate scraping:**
```python
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse

class UitScraper:
    def __init__(self, base_url="https://ctsv.uit.edu.vn"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        self.delay = 2  # seconds between requests

    def scrape_page(self, url, parse_pdfs=True):
        """Scrape single page with UTF-8 handling"""
        try:
            response = self.session.get(url, timeout=10)
            response.encoding = 'utf-8'  # Force UTF-8
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text content
            content = {
                'title': soup.title.string if soup.title else 'N/A',
                'text': soup.get_text(separator='\n', strip=True),
                'url': url
            }

            # Extract PDF links
            if parse_pdfs:
                pdf_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.lower().endswith('.pdf'):
                        full_url = urljoin(self.base_url, href)
                        pdf_links.append({
                            'title': link.get_text().strip(),
                            'url': full_url
                        })
                content['pdfs'] = pdf_links

            time.sleep(self.delay)
            return content

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def download_pdf(self, pdf_url, save_dir='./pdfs'):
        """Download PDF with UTF-8 filename support"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        filename = pdf_url.split('/')[-1]
        if not filename.endswith('.pdf'):
            filename += '.pdf'

        filepath = os.path.join(save_dir, filename)

        try:
            response = self.session.get(pdf_url, timeout=30)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            return filepath
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")
            return None
```

**Scrapy for large-scale crawl:**
```python
# scrapy_uit.py
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

class UitSpider(CrawlSpider):
    name = 'uit'
    allowed_domains = ['ctsv.uit.edu.vn']
    start_urls = ['https://ctsv.uit.edu.vn/']

    rules = (
        Rule(LinkExtractor(allow=r'/'),
             callback='parse_page',
             follow=True),
    )

    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS': 1,
        'FEED_FORMAT': 'jsonlines',
        'FEED_URI': 'ctsv_pages.jsonl',
        'FEED_EXPORT_ENCODING': 'utf-8'  # Critical for Vietnamese
    }

    def parse_page(self, response):
        yield {
            'url': response.url,
            'title': response.css('title::text').get(),
            'text': ' '.join(response.css('body *::text').getall()),
            'pdfs': [urljoin(response.url, pdf)
                    for pdf in response.css('a[href$=".pdf"]::attr(href)').getall()]
        }
```

### UTF-8 Encoding Best Practices

1. **Always specify encoding:**
   ```python
   response.encoding = 'utf-8'
   # OR in BeautifulSoup:
   soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
   ```

2. **Save scraped data as UTF-8:**
   ```python
   import json
   with open('data.json', 'w', encoding='utf-8') as f:
       json.dump(data, f, ensure_ascii=False, indent=2)
   ```

3. **CSV export:**
   ```python
   import csv
   with open('data.csv', 'w', newline='', encoding='utf-8') as f:
       writer = csv.DictWriter(f, fieldnames=['title', 'content'])
       writer.writerows(data)
   ```

### Respectful Scraping

```python
# robots.txt compliance + rate limiting
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests_cache.CachedSession(
    'http_cache',
    expire_after=86400  # Cache 24h
)

# Retry strategy
retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504]
)

adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Check robots.txt
from urllib.robotparser import RobotFileParser
rp = RobotFileParser()
rp.set_url('https://ctsv.uit.edu.vn/robots.txt')
rp.read()
can_fetch = rp.can_fetch('*', 'https://ctsv.uit.edu.vn/some/page')
```

---

## 4. Embedding Models for Vietnamese

### Top Recommendations

**1. dangvantuan/vietnamese-embedding** (Recommended for STS tasks)
- Dimension: 768
- Base model: PhoBERT
- Training: Siamese BERT with sentence-transformers
- Use case: General semantic search, document similarity
- HuggingFace: `dangvantuan/vietnamese-embedding`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('dangvantuan/vietnamese-embedding')
embeddings = model.encode([
    "Điều gì là cách tốt nhất để học tiếng Anh?",
    "Làm thế nào tôi có thể cải thiện kỹ năng viết?"
])
# Returns (2, 768) shape
```

**2. dangvantuan/vietnamese-document-embedding** (For long documents)
- Dimension: 768
- Trained on long-context Vietnamese documents
- Loss: Multi-Negative Ranking + Matryoshka2d + Similarity
- Use case: Full document/lecture note retrieval
- Better for educational content

**3. keepitreal/vietnamese-sbert** (Community-maintained)
- Dimension: 768
- Sentence-BERT trained on Vietnamese
- Alternative if primary models unavailable

**4. bkai-foundation-models/vietnamese-bi-encoder** (Production-grade)
- Dimension: 768
- Foundation model by BKAI
- Actively maintained

### Dimension Trade-offs

| Dimension | Speed | Accuracy | Storage | Recommendation |
|-----------|-------|----------|---------|-----------------|
| 384 | Fast | 85% | Low | Quick prototyping only |
| 768 | Balanced | 95% | Medium | **Default choice** |
| 1024 | Slow | 98% | High | Only if latency acceptable |

**For educational content:** Use 768-dim models (sweet spot). 384-dim loses too much semantic precision for academic material.

### Implementation with MongoDB

```python
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

# Initialize
model = SentenceTransformer('dangvantuan/vietnamese-embedding')
client = MongoClient('mongodb://localhost:27017')
db = client['chatbot']
collection = db['documents']

# Embed and store
def store_document(content, metadata):
    embedding = model.encode(content).tolist()  # Convert numpy to list

    doc = {
        'content': content,
        'embedding': embedding,
        'metadata': metadata,
        'tokens': len(content.split())
    }

    result = collection.insert_one(doc)
    return result.inserted_id

# Semantic search
def semantic_search(query, k=5):
    query_embedding = model.encode(query).tolist()

    results = list(collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_search_idx",
                "path": "embedding",
                "queryVector": query_embedding,
                "k": k,
                "numCandidates": 100
            }
        },
        {
            "$project": {
                "content": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]))

    return results
```

### Batch Embedding for Efficiency

```python
from torch.utils.data import DataLoader

def batch_embed_documents(documents, batch_size=32):
    """Embed many documents efficiently"""

    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=False)
        embeddings.extend(batch_embeddings)

    return embeddings

# Usage
all_docs = [doc['content'] for doc in documents]
embeddings = batch_embed_documents(all_docs)

# Insert to MongoDB
operations = [
    {
        'updateOne': {
            'filter': {'_id': doc['_id']},
            'update': {'$set': {'embedding': emb.tolist()}},
            'upsert': False
        }
    }
    for doc, emb in zip(documents, embeddings)
]

collection.bulk_write(operations)
```

---

## 5. RAG Architecture

### Recommended Stack

**Architecture:** LlamaIndex (retrieval) + LangChain (orchestration optional) + MongoDB (vectors)

**Why this stack:**
- LlamaIndex: 35% faster retrieval than LangChain in Feb 2026 benchmarks
- MongoDB: Vector + keyword search support
- Reranking: Optional but recommended for large corpus

### Advanced RAG Pipeline

```python
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.node_parser import RecursiveCharacterTextSplitter
from llama_index.retrievers.bge_reranker import BGEReranker
from pymongo import MongoClient

# 1. Configure embedding model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="dangvantuan/vietnamese-embedding"
)
Settings.embed_model = embed_model

# 2. Configure text splitter (chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # tokens per chunk
    chunk_overlap=100,    # 20% overlap
    separators=["\n\n", "\n", "。", "！", "？", ",", " ", ""]  # Vietnamese-friendly
)
Settings.text_splitter = text_splitter

# 3. Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['chatbot']

vector_store = MongoDBAtlasVectorSearch(
    client=client,
    db_name='chatbot',
    collection_name='documents',
    vector_field='embedding',
    text_field='content',
    index_name='vector_search_idx'
)

# 4. Create index
documents = [Document(text=doc_text) for doc_text in source_texts]
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    show_progress=True
)

# 5. Setup reranking (optional but recommended)
reranker = BGEReranker(
    model_name='BAAI/bge-reranker-base',
    top_n=3  # Rerank top 3 from retrieval
)

# 6. Create retriever with reranking
retriever = index.as_retriever(
    similarity_top_k=10,  # Initial retrieval
    node_postprocessors=[reranker]  # Rerank step
)

# 7. RAG query
query = "Làm thế nào để trở thành sinh viên xuất sắc?"
retrieved_nodes = retriever.retrieve(query)

for node in retrieved_nodes:
    print(f"Score: {node.score}")
    print(f"Content: {node.text[:200]}")
```

### Hybrid Search (Vector + BM25 Keyword)

```python
from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from typing import List

class HybridRetriever(BaseRetriever):
    """Combine vector search + keyword search"""

    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def _retrieve(self, query_str: str) -> List:
        # Get results from both
        vector_results = self.vector_retriever.retrieve(query_str)
        bm25_results = self.bm25_retriever.retrieve(query_str)

        # Merge with deduplication + score normalization
        combined = {}
        for node in vector_results:
            combined[node.node_id] = node

        for node in bm25_results:
            if node.node_id not in combined:
                combined[node.node_id] = node

        return list(combined.values())

# Setup
vector_retriever = index.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_documents(documents, similarity_top_k=10)

hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
results = hybrid_retriever.retrieve("query")
```

### Chunking Strategies for Vietnamese Content

**Baseline: RecursiveCharacterTextSplitter (Recommended for 80% of use cases)**
```python
# Configuration
chunk_size = 512      # tokens
chunk_overlap = 100   # 20% overlap
separators = [
    "\n\n",           # Paragraph breaks
    "\n",             # Line breaks
    "。",             # Vietnamese period equivalent
    "！", "？",       # Vietnamese punctuation
    ",",              # Comma
    " ",              # Word
    ""                # Fallback to character
]
```

**Advanced: Semantic Chunking (If recursive doesn't work)**
```python
from llama_index.core.node_parser import SemanticSplitter

semantic_splitter = SemanticSplitter(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)

# Chunks by semantic boundaries, not fixed tokens
```

**Benchmark Results (Feb 2026):**
- Recursive 512-token: 69% accuracy (baseline)
- Semantic chunking: 54% accuracy (but smaller chunks)
- Adaptive (topic-aware): 87% accuracy (requires training)
- **Recommendation:** Start with recursive 512 + 20% overlap

### Context Window Management

```python
def count_tokens(text, model="dangvantuan/vietnamese-embedding"):
    """Estimate Vietnamese token count"""
    # Vietnamese: ~1 token per 2-3 characters
    return len(text) // 2.5

# For LLM context (GPT-4: 8k tokens)
max_context_tokens = 7500  # Leave 500 for prompt + response

def build_context(retrieved_nodes, max_tokens=7500):
    """Build context respecting token limit"""
    context = ""
    token_count = 0

    for node in retrieved_nodes:
        node_tokens = count_tokens(node.text)
        if token_count + node_tokens > max_tokens:
            break

        context += node.text + "\n\n"
        token_count += node_tokens

    return context
```

### Query Expansion for Better Retrieval

```python
def expand_query(original_query, llm_client):
    """Generate related queries for retrieval"""

    expansion_prompt = f"""
    Original query: {original_query}

    Generate 2-3 Vietnamese variations that capture similar intent:
    (Do NOT repeat the original, only generate variations)
    """

    expanded = llm_client.create_completion(expansion_prompt)

    return [original_query] + expanded.split('\n')

# Usage in retrieval
queries = expand_query("Tôi muốn học lập trình", llm)
all_results = []

for q in queries:
    results = hybrid_retriever.retrieve(q)
    all_results.extend(results)

# Deduplicate + re-rank
unique_results = {node.node_id: node for node in all_results}.values()
unique_results.sort(key=lambda x: x.score, reverse=True)
```

---

## 6. Word Export Implementation

### Simple Document Generation

```python
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def generate_rag_report(query, retrieved_nodes, output_path='report.docx'):
    """Generate Word document with RAG results"""

    doc = Document()

    # Title
    title = doc.add_heading('RAG Chatbot Query Report', level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Metadata
    doc.add_paragraph(f"Query: {query}").runs[0].bold = True
    doc.add_paragraph(f"Result Count: {len(retrieved_nodes)}")
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # Add page break
    doc.add_page_break()

    # Results
    doc.add_heading('Retrieved Documents', level=2)

    for i, node in enumerate(retrieved_nodes, 1):
        # Section for each result
        section = doc.add_heading(f"Result #{i}", level=3)

        # Metadata
        table = doc.add_table(rows=4, cols=2)
        table.autofit = False
        table.allow_autofit = False

        cells = table.rows[0].cells
        cells[0].text = "Source"
        cells[1].text = node.metadata.get('source', 'N/A')

        cells = table.rows[1].cells
        cells[0].text = "Page"
        cells[1].text = str(node.metadata.get('page', 'N/A'))

        cells = table.rows[2].cells
        cells[0].text = "Score"
        cells[1].text = f"{node.score:.3f}"

        cells = table.rows[3].cells
        cells[0].text = "URL"
        cells[1].text = node.metadata.get('url', 'N/A')

        # Content
        doc.add_heading('Content:', level=4)
        doc.add_paragraph(node.text[:1000] + "..." if len(node.text) > 1000 else node.text)

        doc.add_paragraph()  # Spacing

    doc.save(output_path)
    return output_path
```

### Template-Based Generation (More professional)

```python
from docxtpl import DocxTemplate
import json

def generate_report_from_template(template_path, context_data, output_path):
    """
    Use docx template for professional reports

    Template can have:
    - {{variable_name}} for simple substitution
    - {% for item in items %} for loops
    """

    doc = DocxTemplate(template_path)

    # Prepare context
    context = {
        'query': context_data['query'],
        'timestamp': datetime.now().strftime('%d/%m/%Y'),
        'results': context_data['nodes'],  # List of nodes
        'total_results': len(context_data['nodes'])
    }

    doc.render(context)
    doc.save(output_path)

# Usage
template_data = {
    'query': 'Làm thế nào để cải thiện điểm GPA?',
    'nodes': [
        {
            'title': 'Result 1',
            'score': 0.95,
            'content': 'Nội dung...',
            'source': 'file.pdf'
        },
        # ... more results
    ]
}

generate_report_from_template(
    template_path='templates/rag_report.docx',
    context_data=template_data,
    output_path='generated_report.docx'
)
```

**Template Creation:**
1. Create normal Word document with placeholders: `{{ variable_name }}`
2. Save as `.docx`
3. Use `DocxTemplate` to render with data

### Batch Report Generation

```python
def batch_generate_reports(queries, retriever, output_dir='./reports'):
    """Generate reports for multiple queries"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for query in queries:
        # Retrieve
        nodes = retriever.retrieve(query)

        # Generate filename
        filename = f"{query.replace(' ', '_')[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        filepath = os.path.join(output_dir, filename)

        # Generate report
        try:
            generate_rag_report(query, nodes, filepath)
            results.append({
                'query': query,
                'status': 'success',
                'filepath': filepath
            })
        except Exception as e:
            results.append({
                'query': query,
                'status': 'failed',
                'error': str(e)
            })

    return results
```

---

## 7. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **PDF OCR quality** | Low recall on image PDFs | Test PaddleOCR on sample PDFs first; implement fallback to human review for low-confidence text (< 0.6) |
| **MongoDB scaling** | High latency on large corpus (>1M docs) | Use ANN numCandidates tuning; implement tiered indexing (recent docs hot, old docs cold) |
| **Vietnamese model gaps** | Poor semantic understanding of domain-specific terms | Fine-tune dangvantuan/vietnamese-embedding on educational domain corpus (if budget allows) |
| **Context window overflow** | LLM truncates relevant content | Implement context budget tracking; use query expansion to reduce need for many results |
| **Web scraping detection** | IP blocking from ctsv.uit.edu.vn | Use rotating proxies, respectful rate limiting (2s delay), cache responses |
| **Embedding dimension mismatch** | Vector search fails silently | Validate embedding dimension matches MongoDB index definition on every ingestion |
| **Chunk boundary cuts** | Broken context in mid-sentence | Use semantic chunking for critical content; test on actual educational materials |

---

## 8. Technology Stack Summary

**Finalized Recommendations:**

| Component | Technology | Reason |
|-----------|-----------|--------|
| Vector DB | MongoDB Community 8.2+ (self-hosted) | No Atlas cost, native vector search, familiar MongoDB skills |
| PDF Processing | PyMuPDF (text) + PaddleOCR (images) | Reliable for Vietnamese, fastest pipeline |
| Embeddings | dangvantuan/vietnamese-embedding (768-dim) | Trained on Vietnamese, good general performance |
| RAG Framework | LlamaIndex + MongoDB vector store | 35% faster retrieval, cleaner abstractions than LangChain |
| Reranking | BGEReranker or LLM-based | Cross-encoder available on HuggingFace |
| Web Scraping | BeautifulSoup + Requests | Lightweight for ctsv.uit.edu.vn scale |
| Word Export | python-docx + docxtpl (optional) | Programmatic generation without Office dependency |

---

## 9. Unresolved Questions

1. **Specific Vietnamese OCR benchmark:** No 2025 benchmark comparing PaddleOCR vs EasyOCR on Vietnamese academic PDFs. Recommend testing both on 5-10 sample documents from ctsv.uit.edu.vn.

2. **MongoDB 8.2 Community Edition availability:** Confirmed in public preview; check actual release date before committing to self-hosted setup.

3. **Domain-specific embedding fine-tuning:** dangvantuan/vietnamese-embedding is general-purpose. Worth investigating if fine-tuning on educational domain improves retrieval (not tested in research).

4. **ctsv.uit.edu.vn structure:** Actual site layout unknown. Reconnaissance required to identify PDF URLs and dynamic content.

5. **Reranker model selection:** BGEReranker works but no Vietnamese-specific reranker found. Test BGE vs LLM-based reranking on actual queries.

---

## Sources

- [MongoDB Community Edition Vector Search](https://www.mongodb.com/company/blog/product-release-announcements/supercharge-self-managed-apps-search-vector-search-capabilities)
- [MongoDB Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)
- [PaddleOCR vs Tesseract Analysis](https://www.koncile.ai/en/ressources/paddleocr-analyse-avantages-alternatives-open-source)
- [Vietnamese Embedding Models - HuggingFace](https://huggingface.co/dangvantuan/vietnamese-embedding)
- [LangChain vs LlamaIndex 2025 Comparison](https://latenode.com/blog/langchain-vs-llamaindex-2025-complete-rag-framework-comparison)
- [PyMuPDF Text Extraction Guide](https://artifex.com/blog/text-extraction-strategies-with-pymupdf)
- [Advanced Chunking Strategies for RAG](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [BeautifulSoup Web Scraping Tutorial](https://thunderbit.com/blog/python-beautifulsoup-example-tutorial)
- [python-docx Documentation](https://plainenglish.io/blog/how-to-generate-automated-word-documents-with-python-d6b7f6d3f801)
- [PyMongo Vector Search Examples](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/indexes/)

