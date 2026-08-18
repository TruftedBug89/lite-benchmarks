"""Metadata for benchmarks, categories, icons, and display details."""

from __future__ import annotations

CATEGORY_LABELS: dict[str, str] = {
    "coding": "Coding",
    "science": "Science",
    "math": "Math",
    "knowledge": "Knowledge",
    "instruction": "Instruction",
}

CATEGORY_ICONS: dict[str, str] = {
    "coding": "💻",
    "science": "🔬",
    "math": "📐",
    "knowledge": "📚",
    "instruction": "📋",
}

BENCHMARK_INFO: dict[str, dict[str, str]] = {
    "bigcodebench_hard": {
        "display": "BigCodeBench-Hard",
        "category": "Coding",
        "total": "148",
        "verification": "Python unittest execution (explicit opt-in required)",
        "source": "bigcode/bigcodebench-hard (v0.1.4)",
        "paper": "Zhuo et al. 2024",
        "description": (
            "The hardest 148 practical Python programming tasks from BigCodeBench requiring "
            "deep integration of complex real-world libraries (pandas, numpy, scipy, etc.)."
        ),
    },
    "humanevalplus": {
        "display": "HumanEval+",
        "category": "Coding",
        "total": "164",
        "verification": "Python test execution (explicit opt-in required)",
        "source": "evalplus/humanevalplus",
        "paper": "Chen et al. 2021, augmented by Liu et al. 2023 (EvalPlus)",
        "description": (
            "164 hand-written Python functions with docstrings and rigorously expanded "
            "test cases to catch edge-case bugs and hallucinated solutions."
        ),
    },
    "mbppplus": {
        "display": "MBPP+",
        "category": "Coding",
        "total": "378",
        "verification": "Python test execution (explicit opt-in required)",
        "source": "evalplus/mbppplus",
        "paper": "Austin et al. 2021, augmented by Liu et al. 2023 (EvalPlus)",
        "description": (
            "378 crowd-sourced Python programming problems with heavily augmented "
            "test suites from EvalPlus for deep coverage."
        ),
    },
    "humaneval": {
        "display": "HumanEval",
        "category": "Coding",
        "total": "164",
        "verification": "Python test execution (explicit opt-in required)",
        "source": "evalplus/humanevalplus",
        "paper": "Chen et al. 2021",
        "description": (
            "164 hand-written Python functions with docstrings. The model must "
            "generate a working implementation."
        ),
    },
    "mbpp": {
        "display": "MBPP",
        "category": "Coding",
        "total": "378",
        "verification": "Python test execution (explicit opt-in required)",
        "source": "evalplus/mbppplus",
        "paper": "Austin et al. 2021",
        "description": (
            "378 crowd-sourced Python programming problems designed for entry-level programmers."
        ),
    },
    "bigcodebench": {
        "display": "BigCodeBench",
        "category": "Coding",
        "total": "1,140",
        "verification": "Python unittest execution (explicit opt-in required)",
        "source": "bigcode/bigcodebench (v0.1.4)",
        "paper": "Zhuo et al. 2024",
        "description": (
            "Practical Python programming tasks requiring use of real-world libraries."
        ),
    },
    "gpqa": {
        "display": "GPQA Diamond",
        "category": "Science",
        "total": "198",
        "verification": "Multiple choice (4 options)",
        "source": "nichenshun/gpqa_diamond (community mirror of Idavidrein/gpqa)",
        "paper": "Rein et al. 2023",
        "description": (
            "198 graduate-level questions in physics, chemistry, and biology written "
            "by domain experts. Google-proof questions where non-experts score only 34% with internet."
        ),
    },
    "scibench": {
        "display": "SciBench",
        "category": "Science",
        "total": "692",
        "verification": "Numerical / Formula exact match",
        "source": "xw27/scibench",
        "paper": "Wang et al. 2023",
        "description": (
            "College-level scientific textbook problem solving in physics, chemistry, "
            "and thermodynamics requiring multi-step quantitative calculations."
        ),
    },
    "arc": {
        "display": "ARC-Challenge",
        "category": "Science",
        "total": "1,172",
        "verification": "Multiple choice",
        "source": "allenai/ai2_arc (ARC-Challenge)",
        "paper": "Clark et al. 2018",
        "description": (
            "Grade-school science questions from the AI2 Reasoning Challenge."
        ),
    },
    "gsm8k": {
        "display": "GSM8K",
        "category": "Math",
        "total": "1,319",
        "verification": "Numerical exact match (#### format)",
        "source": "openai/gsm8k (main)",
        "paper": "Cobbe et al. 2021",
        "description": (
            "Grade-school math word problems requiring multi-step arithmetic reasoning."
        ),
    },
    "aime": {
        "display": "AIME 2024/2025",
        "category": "Math",
        "total": "90",
        "verification": "Integer exact match (000-999)",
        "source": "AI-MO/aimo-validation-aime",
        "paper": "MAA AIME Competition Problems",
        "description": (
            "American Invitational Mathematics Examination (AIME) high-school competition math problems. "
            "Premier benchmark for evaluating advanced mathematical reasoning in SOTA AI models."
        ),
    },
    "math_500": {
        "display": "MATH-500",
        "category": "Math",
        "total": "500",
        "verification": "Exact match / \\boxed{} extraction",
        "source": "HuggingFaceH4/MATH-500",
        "paper": "Hendrycks et al. 2021 / Lightman et al. 2023",
        "description": (
            "500 challenging competition math problems (Levels 1 to 5) across algebra, geometry, "
            "number theory, calculus, and probability."
        ),
    },
    "mmlu_pro": {
        "display": "MMLU-Pro",
        "category": "Knowledge",
        "total": "12,032",
        "verification": "Multiple choice (10 options)",
        "source": "TIGER-Lab/MMLU-Pro",
        "paper": "Wang et al. 2024",
        "description": (
            "A harder successor to MMLU with 10 answer choices instead of 4, covering "
            "14 academic disciplines (biology, business, chemistry, computer science, "
            "economics, engineering, health, history, law, math, philosophy, physics, "
            "psychology, other)."
        ),
    },
    "ifeval": {
        "display": "IFEval",
        "category": "Instruction",
        "total": "541",
        "verification": "25 programmatic verifiers (strict)",
        "source": "google/IFEval",
        "paper": "Zhou et al. 2023",
        "description": (
            "Tests whether models follow specific formatting and content instructions "
            "(word counts, paragraph structure, keyword inclusion/exclusion, JSON output, "
            "language constraints, etc.). Each prompt has one or more verifiable "
            "constraints checked by 25 deterministic programmatic verifiers."
        ),
    },
    "supergpqa": {
        "display": "SuperGPQA",
        "category": "Knowledge",
        "total": "26,529 (7,050 hard)",
        "verification": "Multiple choice (up to 10 options)",
        "source": "m-a-p/SuperGPQA",
        "paper": "M-A-P Team 2025",
        "description": (
            "Graduate-level multiple-choice questions across 285 disciplines designed "
            "to evaluate reasoning capabilities at the frontier (hard subset)."
        ),
    },
    "scicode": {
        "display": "SciCode",
        "category": "Coding",
        "total": "65",
        "verification": "Python code execution & unit test assertions",
        "source": "SciCode1/SciCode",
        "paper": "SciCode Team 2024",
        "description": (
            "Research-level scientific Python programming problems across physics, chemistry, "
            "biology, and materials science requiring multi-step numerical algorithms."
        ),
    },
    "tau_bench": {
        "display": "Tau-Bench (Retail)",
        "category": "Instruction",
        "total": "82",
        "verification": "Agentic tool-call function & argument matching",
        "source": "amityco/tau-bench-retail-train-next-action",
        "paper": "Sierra Research 2024",
        "description": (
            "Multi-turn agentic customer service workflows evaluating precise tool choice, "
            "function calling, and user dialogue trajectory control."
        ),
    },
}

import os
import re
import json
import subprocess
from pathlib import Path

_QUANT_REGEX = re.compile(
    r"(?:[_\-\.\/\s]|^)(Q[0-9]+_[A-Z0-9_]+|IQ[0-9]+_[A-Z0-9_]+|Q[0-9]+_[0-9]+|BF16|FP16|F16|F32|AWQ|GPTQ|EXL2|Q[0-9]+_K(?:_[SML])?|Q1_0|Q2_K|Q3_K_[SML]|Q4_K_[MS]|Q4_0|Q4_1|Q5_K_[MS]|Q5_0|Q5_1|Q6_K|Q8_0)(?:[_\-\.\/\s]|$)",
    re.IGNORECASE,
)


def extract_quantization(text: str) -> str | None:
    """Extract quantization tag (e.g. Q4_K_M, Q8_0, FP16) from a model name, ID, or filepath."""
    if not text:
        return None
    m = _QUANT_REGEX.search(text)
    if m:
        return m.group(1).upper()
    return None


def get_lm_studio_catalog() -> list[dict]:
    """Query local LM Studio models if lms CLI or cache directory exists."""
    lms_candidates = [
        Path.home() / ".cache" / "lm-studio" / "bin" / "lms.exe",
        Path.home() / ".cache" / "lm-studio" / "bin" / "lms",
    ]
    for exe in lms_candidates:
        if exe.is_file():
            try:
                res = subprocess.run(
                    [str(exe), "ls", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if res.returncode == 0 and res.stdout:
                    data = json.loads(res.stdout)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
    return []


def get_lm_studio_concrete_configs() -> dict[str, dict]:
    """Parse user-concrete-model-default-config files in LM Studio cache for KV quant, flash attn, and context."""
    results: dict[str, dict] = {}
    config_dir = Path.home() / ".cache" / "lm-studio" / ".internal" / "user-concrete-model-default-config"
    if not config_dir.exists():
        return results

    for root, _, files in os.walk(config_dir):
        for f in files:
            if f.endswith(".json"):
                fp = Path(root) / f
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    fields = data.get("load", {}).get("fields", [])
                    fmap = {item["key"]: item.get("value") for item in fields if isinstance(item, dict) and "key" in item}
                    
                    # Extract KV cache
                    k_cache = fmap.get("llm.load.llama.kCacheQuantizationType")
                    kv_val = None
                    if isinstance(k_cache, dict):
                        if k_cache.get("checked") or k_cache.get("value"):
                            kv_val = str(k_cache.get("value") or "").lower()
                    elif isinstance(k_cache, str):
                        kv_val = k_cache.lower()

                    flash_attn = fmap.get("llm.load.llama.flashAttention")
                    ctx_len = fmap.get("llm.load.contextLength")
                    
                    cfg_dict = {
                        "kv_quant": kv_val,
                        "flash_attention": bool(flash_attn) if flash_attn is not None else None,
                        "context_length": int(ctx_len) if ctx_len else None,
                    }
                    results[f.lower()] = cfg_dict
                    stem = f.lower().replace(".gguf.json", "").replace(".json", "")
                    results[stem] = cfg_dict
                except Exception:
                    pass
    return results


def detect_local_model_metadata(
    model_id: str,
    model_name: str = "",
    api_base: str | None = None,
    explicit_quant: str | None = None,
    explicit_kv_quant: str | None = None,
    explicit_flash_attn: bool | None = None,
) -> dict:
    """Detect or aggregate quantization, KV cache quantization, and Flash Attention for local models."""
    quant = explicit_quant
    kv_quant = explicit_kv_quant
    flash_attn = explicit_flash_attn
    context_length = None

    # 1. Try detecting quant from model_id, model_name, or GGUF filename
    if not quant:
        quant = extract_quantization(model_name) or extract_quantization(model_id)

    # 2. Check LM Studio catalog and concrete configs for matching model
    is_lm_studio = (
        model_id.startswith("lm_studio/")
        or (api_base and "1234" in str(api_base))
        or "lm studio" in model_name.lower()
    )

    if is_lm_studio:
        catalog = get_lm_studio_catalog()
        concrete_cfgs = get_lm_studio_concrete_configs()
        mid_norm = model_id.lower().replace("lm_studio/", "")
        
        # Match catalog
        for item in catalog:
            key = str(item.get("modelKey", "")).lower()
            path = str(item.get("path", "")).lower()
            disp = str(item.get("displayName", "")).lower()
            idx_id = str(item.get("indexedModelIdentifier", "")).lower()
            if (
                key == mid_norm
                or mid_norm in path
                or mid_norm in idx_id
                or key in mid_norm
                or (model_name and disp == model_name.lower())
            ):
                if not quant:
                    q_info = item.get("quantization")
                    if isinstance(q_info, dict) and q_info.get("name"):
                        quant = str(q_info["name"]).upper()
                    elif path:
                        quant = extract_quantization(path)
                
                # Check concrete config match by filename
                fname = Path(path).name.lower()
                for ckey, cval in concrete_cfgs.items():
                    if ckey in fname or fname in ckey or (key and ckey in key):
                        if not kv_quant and cval.get("kv_quant"):
                            kv_quant = cval["kv_quant"]
                        if flash_attn is None and cval.get("flash_attention") is not None:
                            flash_attn = cval["flash_attention"]
                        if not context_length and cval.get("context_length"):
                            context_length = cval["context_length"]
                        break
                break

        # If still missing, check all concrete configs by model_name / id substrings
        if not kv_quant or flash_attn is None:
            for ckey, cval in concrete_cfgs.items():
                if ckey in mid_norm or mid_norm in ckey or (model_name and ckey in model_name.lower()):
                    if not kv_quant and cval.get("kv_quant"):
                        kv_quant = cval["kv_quant"]
                    if flash_attn is None and cval.get("flash_attention") is not None:
                        flash_attn = cval["flash_attention"]
                    if not context_length and cval.get("context_length"):
                        context_length = cval["context_length"]
                    break

    # Defaults
    if not kv_quant and is_lm_studio:
        kv_quant = "q8_0"  # Default to q8_0 for LM Studio users using quantized KV cache
    if flash_attn is None and is_lm_studio:
        flash_attn = True

    return {
        "quantization": quant,
        "kv_quant": kv_quant,
        "flash_attention": flash_attn,
        "context_length": context_length,
    }

