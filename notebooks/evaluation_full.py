"""
Small Model Supremacy — Full Evaluation Pipeline
=================================================
Run on Kaggle with GPU T4 x2. Expected runtime: ~20-30 min.

Produces:
- Schema validity, Field F1, Exact Match with 95% bootstrap CIs
- Per-schema and per-difficulty breakdown
- Latency measurements
- Publication-quality comparison charts
- CSV results for presentation

Usage on Kaggle:
    1. Upload this file + the repo to a Kaggle notebook
    2. Run each section as a cell (separated by # %% markers)
    3. Download results/ folder from Output tab
"""

# %% [markdown]
# # Small Model Supremacy — Full Evaluation Pipeline

# %% Cell 1: Setup
# !pip install peft bitsandbytes accelerate transformers jsonschema matplotlib seaborn pandas numpy --quiet
# !git clone https://github.com/BhavanaR006/small-model-supremacy.git
# %cd small-model-supremacy

import os
import json
import time
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import jsonschema
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

warnings.filterwarnings('ignore')

# Set paths
REPO_ROOT = Path(".")  # Adjust if needed
DATA_DIR = REPO_ROOT / "data"
SCHEMAS_DIR = REPO_ROOT / "schemas"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "final"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
SEED = 42
BOOTSTRAP_ITERATIONS = 1000

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Checkpoint: {CHECKPOINT_DIR}")
print(f"Test set: {DATA_DIR / 'test.jsonl'}")
print("✓ Setup complete")

# %% Cell 2: Load Schemas & Test Data
print("=" * 60)
print("LOADING SCHEMAS & TEST DATA")
print("=" * 60)

# Load schemas
schemas = {}
for schema_file in sorted(SCHEMAS_DIR.glob("*.schema.json")):
    with open(schema_file) as f:
        schema = json.load(f)
    schema_id = schema_file.stem.replace(".schema", "")
    schemas[schema_id] = schema
    print(f"  Loaded schema: {schema_id} ({len(schema.get('properties', {}))} fields)")

# Load test data
test_examples = []
with open(DATA_DIR / "test.jsonl") as f:
    for line in f:
        test_examples.append(json.loads(line.strip()))

print(f"\n  Test examples: {len(test_examples)}")
schema_dist = {}
diff_dist = {}
for ex in test_examples:
    schema_dist[ex['schema_id']] = schema_dist.get(ex['schema_id'], 0) + 1
    diff_dist[ex['difficulty_level']] = diff_dist.get(ex['difficulty_level'], 0) + 1
print(f"  By schema: {schema_dist}")
print(f"  By difficulty: {diff_dist}")

# %% Cell 3: Load Models
print("\n" + "=" * 60)
print("LOADING MODELS")
print("=" * 60)

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
stop_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

# Load base model (for baseline comparison)
print("\n  Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.eval()
print("  ✓ Base model loaded")

# Load fine-tuned model (base + LoRA adapter)
print("  Loading fine-tuned model (LoRA adapter)...")
finetuned_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
finetuned_model = PeftModel.from_pretrained(finetuned_model, str(CHECKPOINT_DIR))
finetuned_model.eval()
print("  ✓ Fine-tuned model loaded")
print(f"\n  Device: {next(finetuned_model.parameters()).device}")

# %% Cell 4: Inference Functions
def format_prompt(text: str, schema_id: str) -> str:
    """Format input text into the chat template used during training."""
    return f"""<|im_start|>system
You are a JSON extraction assistant. You ONLY output valid JSON. No explanations.<|im_end|>
<|im_start|>user
Extract from the following text into the schema "{schema_id}".

Text: {text}<|im_end|>
<|im_start|>assistant
"""


def run_inference(model, prompt: str, max_new_tokens: int = 512) -> tuple[str, float]:
    """Run inference and return (generated_text, latency_ms)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=stop_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start_time) * 1000

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return raw_text.strip(), latency_ms


def parse_json_output(raw: str) -> dict | None:
    """Attempt to parse JSON from model output with repair heuristics."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object boundaries
    try:
        start = raw.index('{')
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == '{':
                depth += 1
            elif raw[i] == '}':
                depth -= 1
            if depth == 0:
                return json.loads(raw[start:i+1])
    except (ValueError, json.JSONDecodeError):
        pass

    # Try fixing common issues: trailing comma, missing closing brace
    cleaned = raw.strip()
    if cleaned.startswith('{') and not cleaned.endswith('}'):
        cleaned += '}'
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    return None

# %% Cell 5: Metrics Functions
def validate_against_schema(output: dict, schema_id: str) -> bool:
    """Check if output conforms to the JSON schema."""
    if schema_id not in schemas:
        return False
    try:
        jsonschema.validate(output, schemas[schema_id])
        return True
    except jsonschema.ValidationError:
        return False


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict for field-level comparison."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            # Flatten list items with index
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]", sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    return dict(items)


def compute_field_f1(predicted: dict, expected: dict) -> dict:
    """Compute field-level precision, recall, F1."""
    pred_flat = flatten_dict(predicted)
    exp_flat = flatten_dict(expected)

    pred_keys = set(pred_flat.keys())
    exp_keys = set(exp_flat.keys())

    if not pred_keys and not exp_keys:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    correct = 0
    for key in pred_keys & exp_keys:
        if str(pred_flat[key]).strip().lower() == str(exp_flat[key]).strip().lower():
            correct += 1

    precision = correct / len(pred_keys) if pred_keys else 0.0
    recall = correct / len(exp_keys) if exp_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_exact_match(predicted: dict, expected: dict) -> bool:
    """Check if predicted matches expected after normalization."""
    def normalize(obj):
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [normalize(x) for x in obj]
        elif isinstance(obj, str):
            return obj.strip().lower()
        return obj
    return normalize(predicted) == normalize(expected)


def bootstrap_ci(values: list, n_iter: int = 1000, confidence: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_iter)])
    alpha = 1.0 - confidence
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))

# %% Cell 6: Run Evaluation
def evaluate_model(model, model_name: str, examples: list) -> pd.DataFrame:
    """Run full evaluation on a model and return per-example results."""
    results = []
    total = len(examples)

    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*60}")

    for i, ex in enumerate(examples):
        prompt = format_prompt(ex['input_text'], ex['schema_id'])
        raw_output, latency_ms = run_inference(model, prompt)
        parsed = parse_json_output(raw_output)

        valid_json = parsed is not None
        schema_valid = validate_against_schema(parsed, ex['schema_id']) if valid_json else False
        field_metrics = compute_field_f1(parsed, ex['expected_output']) if valid_json else {"precision": 0, "recall": 0, "f1": 0}
        exact = compute_exact_match(parsed, ex['expected_output']) if valid_json else False

        results.append({
            "model": model_name,
            "example_idx": i,
            "schema_id": ex['schema_id'],
            "difficulty": ex['difficulty_level'],
            "valid_json": valid_json,
            "schema_valid": schema_valid,
            "field_precision": field_metrics["precision"],
            "field_recall": field_metrics["recall"],
            "field_f1": field_metrics["f1"],
            "exact_match": exact,
            "latency_ms": latency_ms,
            "raw_output_len": len(raw_output),
        })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            valid_so_far = sum(r['schema_valid'] for r in results) / len(results)
            f1_so_far = np.mean([r['field_f1'] for r in results])
            print(f"  [{i+1}/{total}] Schema Valid: {valid_so_far:.1%} | Field F1: {f1_so_far:.3f} | Last latency: {latency_ms:.0f}ms")

    return pd.DataFrame(results)


# Run evaluation on both models
print("\n" + "=" * 60)
print("STARTING FULL EVALUATION (75 examples × 2 models)")
print("=" * 60)
print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

df_finetuned = evaluate_model(finetuned_model, "Qwen2.5-1.5B (QLoRA fine-tuned)", test_examples)
df_base = evaluate_model(base_model, "Qwen2.5-1.5B (base)", test_examples)

# Combine results
df_all = pd.concat([df_finetuned, df_base], ignore_index=True)
print(f"\nCompleted at: {datetime.now().strftime('%H:%M:%S')}")
print(f"Total examples evaluated: {len(df_all)}")

# %% Cell 7: Compute Summary Metrics
print("\n" + "=" * 60)
print("COMPUTING SUMMARY METRICS WITH CONFIDENCE INTERVALS")
print("=" * 60)

def compute_model_summary(df: pd.DataFrame, model_name: str) -> dict:
    """Compute aggregate metrics for a model."""
    model_df = df[df['model'] == model_name]

    schema_validity = model_df['schema_valid'].mean()
    field_f1 = model_df['field_f1'].mean()
    exact_match = model_df['exact_match'].mean()
    valid_json_rate = model_df['valid_json'].mean()
    avg_latency = model_df['latency_ms'].mean()
    median_latency = model_df['latency_ms'].median()

    # Bootstrap CIs
    sv_ci = bootstrap_ci(model_df['schema_valid'].astype(float).tolist(), BOOTSTRAP_ITERATIONS)
    f1_ci = bootstrap_ci(model_df['field_f1'].tolist(), BOOTSTRAP_ITERATIONS)
    em_ci = bootstrap_ci(model_df['exact_match'].astype(float).tolist(), BOOTSTRAP_ITERATIONS)

    return {
        "model": model_name,
        "schema_validity": schema_validity,
        "schema_validity_ci": sv_ci,
        "field_f1": field_f1,
        "field_f1_ci": f1_ci,
        "exact_match": exact_match,
        "exact_match_ci": em_ci,
        "valid_json_rate": valid_json_rate,
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "n_examples": len(model_df),
    }


models = df_all['model'].unique()
summaries = [compute_model_summary(df_all, m) for m in models]

# Print summary table
print(f"\n{'Model':<35} {'Schema Valid':<18} {'Field F1':<18} {'Exact Match':<18} {'Latency (ms)':<12}")
print("-" * 100)
for s in summaries:
    sv = f"{s['schema_validity']:.1%} [{s['schema_validity_ci'][0]:.1%}-{s['schema_validity_ci'][1]:.1%}]"
    f1 = f"{s['field_f1']:.3f} [{s['field_f1_ci'][0]:.3f}-{s['field_f1_ci'][1]:.3f}]"
    em = f"{s['exact_match']:.1%} [{s['exact_match_ci'][0]:.1%}-{s['exact_match_ci'][1]:.1%}]"
    lat = f"{s['avg_latency_ms']:.0f}"
    print(f"{s['model']:<35} {sv:<18} {f1:<18} {em:<18} {lat:<12}")

# %% Cell 8: Per-Schema & Per-Difficulty Breakdown
print("\n" + "=" * 60)
print("PER-SCHEMA BREAKDOWN")
print("=" * 60)

for schema_id in sorted(schemas.keys()):
    print(f"\n  Schema: {schema_id}")
    print(f"  {'Model':<35} {'Schema Valid':<14} {'Field F1':<10} {'Exact Match':<12}")
    print(f"  {'-'*70}")
    for model_name in models:
        subset = df_all[(df_all['model'] == model_name) & (df_all['schema_id'] == schema_id)]
        if len(subset) == 0:
            continue
        sv = subset['schema_valid'].mean()
        f1 = subset['field_f1'].mean()
        em = subset['exact_match'].mean()
        print(f"  {model_name:<35} {sv:.1%}{'':<8} {f1:.3f}{'':<5} {em:.1%}")

print("\n" + "=" * 60)
print("PER-DIFFICULTY BREAKDOWN")
print("=" * 60)

for diff in ['simple', 'medium', 'complex']:
    subset_check = df_all[df_all['difficulty'] == diff]
    if len(subset_check) == 0:
        continue
    print(f"\n  Difficulty: {diff}")
    print(f"  {'Model':<35} {'Schema Valid':<14} {'Field F1':<10} {'Exact Match':<12}")
    print(f"  {'-'*70}")
    for model_name in models:
        subset = df_all[(df_all['model'] == model_name) & (df_all['difficulty'] == diff)]
        if len(subset) == 0:
            continue
        sv = subset['schema_valid'].mean()
        f1 = subset['field_f1'].mean()
        em = subset['exact_match'].mean()
        print(f"  {model_name:<35} {sv:.1%}{'':<8} {f1:.3f}{'':<5} {em:.1%}")

# %% Cell 9: Publication-Quality Charts
print("\n" + "=" * 60)
print("GENERATING CHARTS")
print("=" * 60)

sns.set_style("whitegrid")
sns.set_context("talk")

# --- Chart 1: Grouped Bar Chart — Model Metrics Comparison ---
fig, ax = plt.subplots(figsize=(12, 7))

metric_names = ["Schema Validity", "Field F1", "Exact Match"]
x = np.arange(len(metric_names))
bar_width = 0.35

colors = ['#2196F3', '#FF9800']  # Blue for fine-tuned, orange for base

for i, s in enumerate(summaries):
    values = [s['schema_validity'], s['field_f1'], s['exact_match']]
    cis = [s['schema_validity_ci'], s['field_f1_ci'], s['exact_match_ci']]
    errors_low = [v - ci[0] for v, ci in zip(values, cis)]
    errors_high = [ci[1] - v for v, ci in zip(values, cis)]

    bars = ax.bar(x + i * bar_width, values, bar_width,
                  label=s['model'], color=colors[i], edgecolor='white',
                  linewidth=0.8, yerr=[errors_low, errors_high],
                  capsize=5, error_kw={'linewidth': 1.5})

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}' if val < 1 else f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('')
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Performance Comparison\n(with 95% Bootstrap Confidence Intervals)', fontsize=14, fontweight='bold')
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=11)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✓ Saved: metrics_comparison.png")

# --- Chart 2: Per-Schema Heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, model_name in enumerate(models):
    model_df = df_all[df_all['model'] == model_name]
    pivot_data = model_df.groupby('schema_id').agg({
        'schema_valid': 'mean',
        'field_f1': 'mean',
        'exact_match': 'mean',
    }).rename(columns={
        'schema_valid': 'Schema Validity',
        'field_f1': 'Field F1',
        'exact_match': 'Exact Match',
    })

    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=axes[idx], linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('')

plt.suptitle('Per-Schema Performance Heatmap', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "per_schema_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✓ Saved: per_schema_heatmap.png")

# --- Chart 3: Latency Comparison ---
fig, ax = plt.subplots(figsize=(10, 6))

latency_data = df_all.groupby('model')['latency_ms'].agg(['mean', 'median', 'std']).reset_index()

bars = ax.bar(range(len(latency_data)), latency_data['mean'],
              yerr=latency_data['std'], capsize=8,
              color=colors[:len(latency_data)], edgecolor='white', linewidth=0.8,
              error_kw={'linewidth': 1.5})

for bar, mean_val, med_val in zip(bars, latency_data['mean'], latency_data['median']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + latency_data['std'].max() * 0.1,
            f'Mean: {mean_val:.0f}ms\nMedian: {med_val:.0f}ms',
            ha='center', va='bottom', fontsize=10)

ax.set_xticks(range(len(latency_data)))
ax.set_xticklabels(latency_data['model'], fontsize=11)
ax.set_ylabel('Latency (ms)', fontsize=13)
ax.set_title('Inference Latency Comparison\n(mean ± std)', fontsize=14, fontweight='bold')

plt.tight_layout()
fig.savefig(RESULTS_DIR / "latency_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✓ Saved: latency_comparison.png")

# --- Chart 4: Per-Difficulty Performance (Fine-tuned model) ---
fig, ax = plt.subplots(figsize=(10, 6))

ft_df = df_all[df_all['model'] == "Qwen2.5-1.5B (QLoRA fine-tuned)"]
diff_order = ['simple', 'medium', 'complex']
diff_metrics = ft_df.groupby('difficulty').agg({
    'schema_valid': 'mean',
    'field_f1': 'mean',
    'exact_match': 'mean',
}).reindex(diff_order)

x = np.arange(len(diff_order))
width = 0.25
metric_colors = ['#4CAF50', '#2196F3', '#FF5722']
metric_labels = ['Schema Validity', 'Field F1', 'Exact Match']

for i, (col, color, label) in enumerate(zip(
    ['schema_valid', 'field_f1', 'exact_match'], metric_colors, metric_labels)):
    bars = ax.bar(x + i * width, diff_metrics[col], width,
                  label=label, color=color, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, diff_metrics[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Difficulty Level', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Fine-tuned Model: Performance by Difficulty Level', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([d.capitalize() for d in diff_order], fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "difficulty_breakdown.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✓ Saved: difficulty_breakdown.png")

# --- Chart 5: Radar/Spider Plot ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

categories = ['Schema\nValidity', 'Field F1', 'Exact\nMatch', 'Valid\nJSON', 'Speed\nScore']
n_cats = len(categories)
angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
angles += angles[:1]  # Close the polygon

for i, s in enumerate(summaries):
    # Normalize speed: invert so faster = higher score (cap at 1.0)
    speed_score = min(1.0, 100.0 / max(s['avg_latency_ms'], 1))
    values = [s['schema_validity'], s['field_f1'], s['exact_match'],
              s['valid_json_rate'], speed_score]
    values += values[:1]  # Close

    ax.plot(angles, values, 'o-', linewidth=2, label=s['model'], color=colors[i])
    ax.fill(angles, values, alpha=0.15, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_title('Model Capability Radar', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=10)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "radar_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✓ Saved: radar_comparison.png")

# %% Cell 10: Save CSV Results & Summary Report
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save detailed per-example results
csv_path = RESULTS_DIR / "evaluation_results.csv"
df_all.to_csv(csv_path, index=False)
print(f"  ✓ Saved: {csv_path} ({len(df_all)} rows)")

# Save summary metrics
summary_df = pd.DataFrame(summaries)
summary_df['schema_validity_ci_lower'] = summary_df['schema_validity_ci'].apply(lambda x: x[0])
summary_df['schema_validity_ci_upper'] = summary_df['schema_validity_ci'].apply(lambda x: x[1])
summary_df['field_f1_ci_lower'] = summary_df['field_f1_ci'].apply(lambda x: x[0])
summary_df['field_f1_ci_upper'] = summary_df['field_f1_ci'].apply(lambda x: x[1])
summary_df['exact_match_ci_lower'] = summary_df['exact_match_ci'].apply(lambda x: x[0])
summary_df['exact_match_ci_upper'] = summary_df['exact_match_ci'].apply(lambda x: x[1])
summary_df = summary_df.drop(columns=['schema_validity_ci', 'field_f1_ci', 'exact_match_ci'])

summary_csv_path = RESULTS_DIR / "summary_metrics.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"  ✓ Saved: {summary_csv_path}")

# Save per-schema breakdown
schema_breakdown = df_all.groupby(['model', 'schema_id']).agg({
    'schema_valid': 'mean',
    'field_f1': 'mean',
    'exact_match': 'mean',
    'latency_ms': 'mean',
    'valid_json': 'mean',
}).reset_index()
schema_csv_path = RESULTS_DIR / "per_schema_results.csv"
schema_breakdown.to_csv(schema_csv_path, index=False)
print(f"  ✓ Saved: {schema_csv_path}")

# Save per-difficulty breakdown
diff_breakdown = df_all.groupby(['model', 'difficulty']).agg({
    'schema_valid': 'mean',
    'field_f1': 'mean',
    'exact_match': 'mean',
    'latency_ms': 'mean',
}).reset_index()
diff_csv_path = RESULTS_DIR / "per_difficulty_results.csv"
diff_breakdown.to_csv(diff_csv_path, index=False)
print(f"  ✓ Saved: {diff_csv_path}")

# Generate a markdown summary report
report_lines = [
    "# Evaluation Report — Small Model Supremacy",
    "",
    f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    f"**Test set:** {len(test_examples)} examples across {len(schemas)} schemas",
    f"**Bootstrap iterations:** {BOOTSTRAP_ITERATIONS}",
    "",
    "## Overall Results",
    "",
    "| Model | Schema Validity | Field F1 | Exact Match | Avg Latency |",
    "|-------|:-:|:-:|:-:|:-:|",
]

for s in summaries:
    sv = f"{s['schema_validity']:.1%} ({s['schema_validity_ci'][0]:.1%}–{s['schema_validity_ci'][1]:.1%})"
    f1 = f"{s['field_f1']:.3f} ({s['field_f1_ci'][0]:.3f}–{s['field_f1_ci'][1]:.3f})"
    em = f"{s['exact_match']:.1%} ({s['exact_match_ci'][0]:.1%}–{s['exact_match_ci'][1]:.1%})"
    lat = f"{s['avg_latency_ms']:.0f}ms"
    report_lines.append(f"| **{s['model']}** | {sv} | {f1} | {em} | {lat} |")

report_lines += [
    "",
    "## Per-Schema Results",
    "",
    "| Model | Schema | Schema Validity | Field F1 | Exact Match |",
    "|-------|--------|:-:|:-:|:-:|",
]

for _, row in schema_breakdown.iterrows():
    report_lines.append(
        f"| {row['model']} | {row['schema_id']} | {row['schema_valid']:.1%} | {row['field_f1']:.3f} | {row['exact_match']:.1%} |"
    )

report_lines += [
    "",
    "## Per-Difficulty Results",
    "",
    "| Model | Difficulty | Schema Validity | Field F1 | Exact Match |",
    "|-------|-----------|:-:|:-:|:-:|",
]

for _, row in diff_breakdown.iterrows():
    report_lines.append(
        f"| {row['model']} | {row['difficulty']} | {row['schema_valid']:.1%} | {row['field_f1']:.3f} | {row['exact_match']:.1%} |"
    )

report_lines += [
    "",
    "## Charts",
    "",
    "- `metrics_comparison.png` — Overall metrics comparison with CIs",
    "- `per_schema_heatmap.png` — Per-schema performance heatmap",
    "- `latency_comparison.png` — Inference latency comparison",
    "- `difficulty_breakdown.png` — Performance by difficulty level",
    "- `radar_comparison.png` — Multi-dimensional capability radar",
    "",
    "## Configuration",
    "",
    f"- **Base model:** {BASE_MODEL_NAME}",
    f"- **Adapter:** QLoRA (r=32, alpha=64, 5 epochs on 1500 examples)",
    f"- **Target modules:** q_proj, k_proj, v_proj, o_proj",
    f"- **Quantization:** 4-bit NF4 with double quantization",
    f"- **Inference:** Temperature 0, deterministic (do_sample=False)",
    "",
]

report_path = RESULTS_DIR / "EVALUATION_REPORT.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"  ✓ Saved: {report_path}")

# %% Cell 11: Package Results for Download
print("\n" + "=" * 60)
print("PACKAGING RESULTS")
print("=" * 60)

import shutil

# Create a zip of results
zip_path = shutil.make_archive('/kaggle/working/results', 'zip', str(RESULTS_DIR))
print(f"  ✓ Results packaged: {zip_path}")

# Final summary
print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print(f"\nResults directory contents:")
for f in sorted(RESULTS_DIR.iterdir()):
    if f.name == '.gitkeep':
        continue
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<35} ({size_kb:.1f} KB)")

print(f"\n{'='*60}")
print("KEY FINDINGS")
print(f"{'='*60}")
ft_summary = summaries[0]
base_summary = summaries[1]
print(f"\n  Fine-tuned model improvement over base:")
print(f"    Schema Validity: {ft_summary['schema_validity']:.1%} vs {base_summary['schema_validity']:.1%} (+{(ft_summary['schema_validity'] - base_summary['schema_validity'])*100:.1f}pp)")
print(f"    Field F1:        {ft_summary['field_f1']:.3f} vs {base_summary['field_f1']:.3f} (+{(ft_summary['field_f1'] - base_summary['field_f1']):.3f})")
print(f"    Exact Match:     {ft_summary['exact_match']:.1%} vs {base_summary['exact_match']:.1%} (+{(ft_summary['exact_match'] - base_summary['exact_match'])*100:.1f}pp)")
print(f"\n  Download 'results.zip' from the Output tab.")
print("=" * 60)
