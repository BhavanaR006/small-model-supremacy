"""Upload model card to HuggingFace."""
from huggingface_hub import HfApi

api = HfApi()

model_card = """---
license: mit
base_model: Qwen/Qwen2.5-1.5B
tags:
- peft
- lora
- qlora
- structured-extraction
- json-extraction
- qwen2.5
library_name: peft
pipeline_tag: text-generation
---

# Small Model Supremacy — Qwen2.5-1.5B QLoRA Adapter

A fine-tuned QLoRA adapter that enables Qwen2.5-1.5B to outperform frontier models (GPT-4o, Claude 3.5 Sonnet) on structured JSON extraction from unstructured text.

## Model Details

- **Base model:** Qwen/Qwen2.5-1.5B
- **Method:** QLoRA (4-bit NF4 quantization + LoRA)
- **LoRA config:** r=32, alpha=64, target_modules=[q_proj, k_proj, v_proj, o_proj]
- **Training:** 5 epochs on 1500 synthetic examples (3 schema types)
- **Task:** Structured data extraction from natural language to JSON

## Supported Schemas

1. **conference_talk_simple** — Extract speaker, topic, conference, location
2. **product_listing_medium** — Extract product details with nested pricing/dimensions
3. **scientific_paper_complex** — Extract paper metadata with authors, findings, datasets

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "BhavanaR006/small-model-supremacy-qwen2.5-1.5b")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

prompt = '''<|im_start|>system
You are a JSON extraction assistant. You ONLY output valid JSON. No explanations.<|im_end|>
<|im_start|>user
Extract from the following text into the schema "conference_talk_simple".

Text: Dr. Sarah Chen presented quantum error correction at NeurIPS 2025 in Vancouver.<|im_end|>
<|im_start|>assistant
'''

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
stop_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, eos_token_id=stop_id)
result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)
# {"speaker_name": "Dr. Sarah Chen", "topic": "quantum error correction", "conference": "NeurIPS 2025", "location": "Vancouver"}
```

## Training Details

- **Hardware:** Kaggle T4 GPU (16GB VRAM)
- **Training time:** ~2-3 hours
- **Trainable parameters:** 8.7M / 1.55B total (0.56%)
- **Cost:** $0 (free Kaggle GPU)

## Project

Full source code, training pipeline, and evaluation harness: [GitHub](https://github.com/BhavanaR006/small-model-supremacy)
"""

api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id="BhavanaR006/small-model-supremacy-qwen2.5-1.5b",
    repo_type="model",
    commit_message="Add model card",
)
print("Model card uploaded!")
