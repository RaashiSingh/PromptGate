
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json
import os

import yaml

from .runner import load_cases, run_suite
from .compare import compare
from .report import write_html, write_markdown

from .cli import (
    generate_cache_key,
    load_cached_baseline,
    save_cached_baseline,
)


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

LIVE_RESULTS = ROOT / "results" / "live_runs"


# ==========================================
# LIVE EVALUATION SERVICE
# ==========================================

def evaluate_prompts(
    baseline_prompt,
    candidate_prompt,
    limit=3,
    repeats=1,
    use_cache=True,
    critical_only=False,
    progress_callback=None,
):

    def notify(progress, message):
        if progress_callback is not None:
            progress_callback(progress, message)

    # --------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------

    if not baseline_prompt.strip():
        raise ValueError("Baseline prompt cannot be empty.")

    if not candidate_prompt.strip():
        raise ValueError("Candidate prompt cannot be empty.")

    if repeats < 1:
        raise ValueError("Repeats must be at least 1.")

    if limit is not None and limit < 1:
        raise ValueError("Case limit must be at least 1.")

    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY is not configured.")

    if not os.environ.get("LLM_BASE_URL"):
        raise RuntimeError("LLM_BASE_URL is not configured.")

    # --------------------------------------
    # LOAD CONFIGURATION
    # --------------------------------------

    notify(0.05, "Loading evaluation configuration...")

    with open(
        ROOT / "promptgate.yaml",
        encoding="utf-8"
    ) as file:
        cfg = yaml.safe_load(file)

    cases = load_cases(
        ROOT / "testset" / "cases.jsonl"
    )

    if critical_only:
        critical_group = cfg.get(
            "critical_group", "critical"
        )

        cases = [
            case for case in cases
            if case["group"] == critical_group
        ]

    if limit is not None:
        cases = cases[:limit]

    if not cases:
        raise ValueError("No evaluation cases found.")

    case_ids = [case["id"] for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Evaluation case IDs must be unique."
        )

    model = cfg["model"]
    temperature = cfg.get("temperature", 0.0)
    workers = cfg.get("workers", 2)

    # --------------------------------------
    # CHECK BASELINE CACHE
    # --------------------------------------

    notify(0.15, "Checking baseline cache...")

    base = None
    cache_key = None
    cache_warning = None

    if use_cache:

        cache_key = generate_cache_key(
            baseline_prompt,
            cases,
            model,
            repeats,
            temperature,
            workers,
        )

        base = load_cached_baseline(
            cache_key,
            cases,
            repeats,
        )

    cache_hit = base is not None

    calls_per_suite = len(cases) * repeats

    planned_calls = (
        calls_per_suite
        if cache_hit
        else calls_per_suite * 2
    )

    saved_calls = (
        calls_per_suite if cache_hit else 0
    )

    # --------------------------------------
    # EVALUATE BASELINE
    # --------------------------------------

    if cache_hit:

        notify(
            0.40,
            "Baseline cache hit. Reusing evaluation..."
        )

    else:

        notify(
            0.25,
            "Running baseline prompt..."
        )

        base = run_suite(
            baseline_prompt,
            cases,
            model,
            repeats=repeats,
            temperature=temperature,
            workers=workers,
        )

        if use_cache:
            try:
                save_cached_baseline(
                    cache_key,
                    base,
                    cases,
                    repeats,
                )
            except OSError as error:
                cache_warning = str(error)

    # --------------------------------------
    # EVALUATE CANDIDATE
    # --------------------------------------

    notify(0.50, "Evaluating candidate prompt...")

    cand = run_suite(
        candidate_prompt,
        cases,
        model,
        repeats=repeats,
        temperature=temperature,
        workers=workers,
    )

    # --------------------------------------
    # COMPARE EVALUATIONS
    # --------------------------------------

    notify(0.80, "Detecting prompt regressions...")

    result = compare(
        base,
        cand,
        cases,
        cfg,
    )

    # --------------------------------------
    # CREATE UNIQUE EXPERIMENT ID
    # --------------------------------------

    timestamp = datetime.now(timezone.utc)

    run_id = (
        timestamp.strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:8]
    )

    output_dir = LIVE_RESULTS / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------
    # BUILD COMPLETE EVALUATION RECORD
    # --------------------------------------

    payload = {
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "source": "live_prompt_lab",

        "configuration": {
            "model": model,
            "temperature": temperature,
            "workers": workers,
            "cases": len(cases),
            "repeats": repeats,
            "critical_only": critical_only,
            "tolerance": cfg["tolerance"],
            "group_tolerance": cfg["group_tolerance"],
            "critical_group": cfg.get(
                "critical_group", "critical"
            ),
        },

        "comparison": result,

        "baseline_prompt": baseline_prompt,
        "candidate_prompt": candidate_prompt,

        "cases": cases,

        "baseline_runs": base,
        "candidate_runs": cand,

        "cache": {
            "enabled": use_cache,
            "baseline_cache_hit": cache_hit,
            "cache_key": cache_key,
            "planned_api_calls": planned_calls,
            "api_calls_saved": saved_calls,
            "warning": cache_warning,
        },
    }

    # --------------------------------------
    # SAVE REPORTS
    # --------------------------------------

    notify(0.90, "Generating evaluation reports...")

    (output_dir / "comparison.json").write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_html(
        str(output_dir / "report.html"),
        result,
        cases,
        base,
        cand,
    )

    write_markdown(
        str(output_dir / "report.md"),
        result,
        cases,
        base,
        cand,
    )

    notify(1.0, "Evaluation complete.")

    return payload
