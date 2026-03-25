# Vietnamese LLM Fine-Tuning Research Report
**Date:** 2026-03-21
**Focus:** Open-source LLMs + Colab fine-tuning setup for educational chatbot

---

## Executive Summary

- **Recommended LLM:** Qwen2.5-7B (multilingual, Vietnamese support, 7B fits Colab with QLoRA)
- **Fine-tuning approach:** QLoRA + Unsloth on Colab T4 (8-10GB VRAM) or LoRA on A100 (40GB+)
- **Dataset requirement:** 5K-10K high-quality instruction pairs; quality > quantity
- **Deployment:** Export to GGUF → Ollama (local) or llama.cpp + FastAPI (production)
- **Timeline:** Notebook fine-tuning 2-4 hours (T4), inference optimization <1 hour

---

## 1. Recommended LLM: Qwen2.5-7B

### Why Qwen2.5-7B over alternatives?

| Model | Vietnamese Support | Model Size | Colab T4 Fit | Licensing | Notes |
|-------|-------------------|-----------|-------------|-----------|-------|
| **Qwen2.5-7B** | ✓ Excellent | 7B | ✓ Yes (QLoRA) | Apache 2.0 | Best multilingual, strong instruction-following |
| LLaMA-3.2-7B | ✓ Good | 7B | ✓ Yes (QLoRA) | Llama 2 License | Requires QLoRA for T4 |
| Gemma-2-9B | ✓ Fair | 9B | ✗ Tight (QLoRA needed) | Apache 2.0 | Larger, pushes Colab limits |
| Mistral-7B | ~ Limited | 7B | ✓ Yes (QLoRA) | Apache 2.0 | Less Vietnamese optimization |
| VinaLLaMA-7B | ✓ Excellent | 7B | ✓ Yes (QLoRA) | Apache 2.0 | Vietnamese-native, uses LLaMA-2 base |
| PhoGPT-4B | ✓ Excellent | 4B | ✓ Yes (LoRA) | Research license | Smaller, slower |

**Verdict:**
- **First choice:** `Qwen2.5-7B` (29+ languages, instruction-optimized, proven multilingual)
- **Alternative (Vietnamese-native):** `VinaLLaMA-7B` (built on LLaMA-2 + 800B Vietnamese tokens, state-of-art on VLSP/VMLU)
- **Conservative choice:** `LLaMA-3.2-3B` (smaller footprint, easier on memory, acceptable quality)

**Rationale:** Qwen2.5 balances Vietnamese support, general knowledge, and Colab compatibility. If Vietnamese-specific tuning is critical, VinaLLaMA is worth testing despite slightly older base model.

---

## 2. Fine-Tuning Setup for Colab (T4 Free Tier + Pro)

### QLoRA vs LoRA Decision Matrix

| Aspect | QLoRA (T4 16GB) | LoRA (A100 40GB) |
|--------|--------|--------|
| Memory for 7B model | 8-10GB | 20-24GB |
| 4-bit quantization | Yes | No |
| Speed vs Full fine-tune | ~60% faster | ~40% faster |
| Quality recovery | 80-90% | 90-95% |
| Batch size (seq=2048) | 1-2 | 4-8 |
| Cost (Colab free) | ✓ Runs | ✗ OOM |
| **Recommendation** | **Use for free T4** | **Use for Pro A100** |

### Recommended Tech Stack

#### For Free T4 (Recommended)
```
Primary: Unsloth (2.1x faster, 60% less memory)
  └─ TRL (training loop)
  └─ PEFT/bitsandbytes (QLoRA backend)
  └─ Transformers (model loading)

Secondary: LLaMA Factory
  └─ YAML-based config (easier than code)
  └─ Built-in Unsloth support
```

#### For Colab Pro (A100 40GB)
```
Primary: Unsloth + LoRA (skip 4-bit quantization)
  └─ Standard LoRA rank=32-64
  └─ Batch size 4-6
```

### Colab Notebook Structure

**Phase 1: Setup (5-10 min)**
```python
# Install with Unsloth support
!pip install unsloth[colab-new] xformers transformers bitsandbytes peft trl accelerate

# Choose model
from unsloth import fastlanguage_model, get_chat_template

model_name = "Qwen/Qwen2.5-7B"
# or "meta-llama/Llama-3.2-7B-instruct"
# or "VinAIResearch/VinaLLaMA-7B-chat"

model, tokenizer = fastlanguage_model.get_peft_model(
    model_name,
    max_seq_length=2048,
    dtype="auto",
    load_in_4bit=True,  # QLoRA: 4-bit quantization
)
```

**Phase 2: Data Loading (5 min)**
```python
# Format: ChatML or Alpaca
# Input: system_prompt, instruction, input, output
# Load from HF dataset or local JSON

from datasets import load_dataset

dataset = load_dataset("json", data_files="training_data.jsonl")
# or create custom: [{"prompt": "...", "completion": "..."}]
```

**Phase 3: Training (120-240 min on T4)**
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,  # For T4, avoid packing
    args=TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=2,  # T4 constraint
        gradient_accumulation_steps=4,  # Effective batch = 8
        warmup_steps=10,
        weight_decay=0.01,
        learning_rate=5e-4,  # Conservative for QLoRA
        fp16=True,
        optim="paged_adamw_32bit",  # Memory-efficient
        output_dir="./outputs",
        logging_steps=50,
        save_steps=100,
    ),
)

trainer.train()
```

**Phase 4: Export to GGUF (10-15 min)**
```python
from unsloth import unsloth_to_gguf

model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
# Output: model/GGUF/model-unsloth.gguf (~5GB for 7B model)
```

### VRAM Requirements Breakdown (Qwen2.5-7B)

| Component | QLoRA (4-bit) | Full Precision |
|-----------|---------------|-----------------|
| Model weights (4-bit) | 1.75GB | 14GB |
| Optimizer state (AdamW) | 2GB | 28GB |
| Activations + gradients | 3.5GB | 14GB |
| Overhead | 1GB | 1GB |
| **Total** | **8.25GB** | **57GB** |
| Colab T4 headroom | ✓ 7.75GB free | ✗ -41GB (OOM) |

---

## 3. Training Data Preparation Pipeline

### Format Strategy: ChatML (Recommended)

**Why ChatML over Alpaca:**
- More flexible for multi-turn conversations
- Native support in Transformers
- Better for educational Q&A with context

**ChatML JSON structure:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a Vietnamese biology tutor..."},
    {"role": "user", "content": "Quá trình quang hợp là gì?"},
    {"role": "assistant", "content": "Quang hợp là quá trình mà các cây xanh..."}
  ]
}
```

**Alternative: Alpaca format** (simpler for single-turn)
```json
{
  "instruction": "Giải thích quang hợp",
  "input": "",
  "output": "Quang hợp là quá trình..."
}
```

### Data Generation Pipeline for Educational Content

**Step 1: Scrape/Collect Educational Content**
- Vietnamese educational websites, textbooks, Q&A forums
- Extract topics, key concepts, learning objectives
- Format: Raw text → structured knowledge pairs

**Step 2: Generate Instruction-Response Pairs**
```python
# Semi-automated approach:
# 1. Manual: Create 50-100 seed pairs (high quality)
# 2. Template-based expansion: Use 20 prompt templates
#    - "Explain [topic] to a student"
#    - "What are common misconceptions about [topic]?"
#    - "Provide an example of [concept]"
# 3. Few-shot GPT-3.5/Claude: Expand to 5K pairs (careful: verify output)

# Example template:
templates = [
    "Giải thích {topic} cho một học sinh lớp {grade}",
    "So sánh giữa {concept1} và {concept2}",
    "{concept} được ứng dụng như thế nào trong thực tế?",
    "Điểm chung giữa {topic1} và {topic2} là gì?"
]
```

**Step 3: Quality Control**
- Manual review: First 500 pairs to verify format/language
- Automated filters:
  - Remove duplicates/near-duplicates
  - Check Vietnamese language correctness (length >10 tokens, <500 tokens)
  - Verify instruction-response relevance
- Domain expert review: 20% sample for Vietnamese-specific education jargon

**Step 4: Format and Upload**
```python
# Convert to training format
import json

training_data = []
for topic, qa_pairs in content.items():
    for q, a in qa_pairs:
        training_data.append({
            "messages": [
                {"role": "system", "content": "Bạn là một gia sư giáo dục tuyệt vời."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

with open("training_data.jsonl", "w", encoding="utf-8") as f:
    for line in training_data:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
```

### Minimum Dataset Size Recommendations

| Dataset Size | Quality Level | Expected Output |
|-------------|---------------|-----------------|
| 1K-2K pairs | Excellent (hand-curated) | Good tuning, niche domain |
| 5K-10K pairs | Good (mixed sources) | **Recommended for educational chatbot** |
| 20K+ pairs | Mixed (includes auto-generated) | Diminishing returns unless high-quality |

**For this project:** Target 5K-7K high-quality Vietnamese instruction pairs. Quality > quantity; 5K excellent pairs > 50K mediocre pairs.

---

## 4. Model Deployment Post-Fine-Tuning

### Export Workflow

**Step 1: Convert to GGUF** (in Colab notebook)
```python
from unsloth import unsloth_to_gguf

model.save_pretrained_gguf(
    "final_model",
    tokenizer,
    quantization_method="q4_k_m",  # 4-bit quantization
    # q4_k_m: best quality/speed trade-off
    # q8_0: higher quality, larger file
    # f16: no quantization, 14GB file
)
# Output: ~5GB GGUF file
```

### Deployment Options Comparison

| Platform | Setup Time | Memory | Latency | Multi-user | Best For |
|----------|-----------|--------|---------|-----------|----------|
| **Ollama** | 5 min | 6GB | 1-2s | Basic load balancing | Local development, demo |
| **llama.cpp** | 10 min | 4GB (low-rank) | 0.5-1s | Simple HTTP | Edge deployment, inference server |
| **vLLM** | 15 min | 10GB+ | 0.2s | Excellent (PagedAttention) | Production, high throughput |

### Recommended: llama.cpp + FastAPI (Scalable)

**Setup:**
```bash
# 1. Clone and build llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make -j$(nproc)

# 2. Run inference server (OpenAI-compatible API)
./llama-cpp-server \
  --model model-unsloth.gguf \
  --ctx-size 2048 \
  --threads $(nproc) \
  --port 8000

# Server now at: http://localhost:8000/v1/chat/completions
```

**Python FastAPI wrapper:**
```python
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-dummy"  # llama.cpp doesn't require real API key
)

@app.post("/chat")
async def chat(message: str, context: str = ""):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # ignored by llama.cpp
        messages=[
            {"role": "system", "content": "Bạn là gia sư giáo dục."},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return {"response": response.choices[0].message.content}
```

### Local Development Alternative: Ollama

```bash
# 1. Install Ollama (https://ollama.ai)
# 2. Create Modelfile
cat > Modelfile <<EOF
FROM ./model-unsloth.gguf
SYSTEM "Bạn là gia sư giáo dục tuyệt vời"
EOF

# 3. Create and run
ollama create vietnamese-tutor -f Modelfile
ollama run vietnamese-tutor "Quang hợp là gì?"
```

---

## 5. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Colab session timeout (12h free tier)** | Loss of progress, retraining needed | Enable GPU auto-select, save checkpoints every 100 steps, reduce epochs to 1 |
| **OOM on T4 with larger batch size** | Training crash, data loss | Start batch=1, use gradient_accumulation=4, enable gradient_checkpointing |
| **Poor Vietnamese quality from auto-generated data** | Chatbot outputs low-quality Vietnamese | Manual review 20% of pairs, test on native speaker feedback |
| **Catastrophic forgetting** (overfitting to domain) | Loss of general knowledge | Use lower learning rate (5e-4), shorter epochs (1), keep 10-20% general knowledge data |
| **GGUF quantization artifacts** | Inference quality degradation | Test q4_k_m first; if issues, try q8_0; measure perplexity before deployment |
| **Slow inference on CPU** | >3s latency unacceptable for chatbot | Use GPU inference (A100 or RTX 3090) if possible; llama.cpp faster than vLLM for small models |

---

## 6. Recommended Project Timeline

| Phase | Tool | Duration | Hardware | Deliverable |
|-------|------|----------|----------|-------------|
| Setup + data prep | Python/Colab | 1-2 days | CPU | 5K training pairs (JSONL) |
| Fine-tuning | Unsloth + TRL | 2-4 hours | Colab T4 | Checkpoint + LoRA weights |
| Export + test | unsloth_to_gguf | 30 min | Colab | model.gguf (~5GB) |
| Inference setup | llama.cpp | 1 hour | Local machine | HTTP API serving |
| Integration | FastAPI | 2-4 hours | Local | Python chatbot endpoint |

---

## Unresolved Questions

1. **Vietnamese diacritical robustness:** Do Qwen/LLaMA tokenizers handle Vietnamese tone marks equally well across fine-tuning? (Test needed with VinaLLaMA vs Qwen2.5)

2. **VinaLLaMA vs Qwen2.5 for education:** Which performs better on Vietnamese educational QA after fine-tuning? (Requires benchmark on VLSP/VMLU equivalents)

3. **Synthetic data quality threshold:** What's the minimum manual review rate (10% vs 50%) to avoid degradation on auto-generated pairs?

4. **Production inference latency targets:** What's acceptable (<1s, <2s) for your chatbot UX? Affects deployment choice (llama.cpp vs vLLM).

5. **Multi-turn context handling:** Should system prompt include conversation history, and at what max depth before context explosion?

---

## Sources

- [Ultimate Guide - Best Open Source LLM For Vietnamese In 2026](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Vietnamese)
- [QLoRA - How to Fine-Tune an LLM on a Single GPU](https://towardsdatascience.com/qlora-how-to-fine-tune-an-llm-on-a-single-gpu-4e44d6b5be32/)
- [LoRA vs QLoRA: Best AI Model Fine-Tuning Platforms & Tools 2026](https://www.index.dev/blog/top-ai-fine-tuning-tools-lora-vs-qlora-vs-full)
- [Unsloth Documentation - Fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- [Fine-tuning made easy with Unsloth and Colab](https://medium.com/@amrstech/fine-tuning-made-easy-with-unsloth-and-colab-e0993f3f4c07)
- [VinaLLaMA: LLaMA-based Vietnamese Foundation Model](https://arxiv.org/html/2312.11011v1)
- [Towards Comprehensive Vietnamese Retrieval-Augmented Generation and Large Language Models](https://arxiv.org/html/2403.01616v1)
- [Efficient Finetuning Large Language Models For Vietnamese Chatbot](https://arxiv.org/abs/2309.04646)
- [The Rise of GGUF Models: Why They're Changing How We Do Inference](https://www.runpod.io/articles/guides/the-rise-of-gguf-models-why-theyre-changing-inference)
- [Local LLM Hosting: Complete 2025 Guide](https://medium.com/@rosgluk/local-llm-hosting-complete-2025-guide-ollama-vllm-localai-jan-lm-studio-more-f98136ce7e4a)
- [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/html/2308.10792v9)
- [Fine-tuning large language models (LLMs) in 2025](https://www.superannotate.com/blog/llm-fine-tuning)
- [Run llama.cpp Server: OpenAI-Compatible API from GGUF Models 2026](https://markaicode.com/llama-cpp-server-openai-api-gguf/)
- [vLLM or llama.cpp: Choosing the right LLM inference engine](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)
- [Colab GPUs Features & Pricing](http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
- [PhoGPT: Generative Pre-training for Vietnamese](https://github.com/VinAIResearch/PhoGPT)
