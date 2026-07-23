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
