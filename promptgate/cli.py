
import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from .runner import load_cases, run_suite
from .compare import compare
from .report import write_html, write_markdown


# ==========================================
# CACHE CONFIGURATION
# ==========================================

CACHE_DIR = Path("results/cache/baselines")
CACHE_VERSION = 1

PACKAGE_DIR = Path(__file__).resolve().parent


# ==========================================
# GENERATE BASELINE CACHE KEY
# ==========================================

def generate_cache_key(
    prompt,
    cases,
    model,
    repeats,
    temperature,
    workers
):
    """Generate a fingerprint for the baseline evaluation."""

    code_hashes = {}

    for filename in (
        "llm.py",
        "runner.py",
        "scoring.py",
    ):
        path = PACKAGE_DIR / filename

        code_hashes[filename] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    payload = {
        "cache_version": CACHE_VERSION,
        "prompt": prompt,
        "cases": cases,
        "model": model,
        "repeats": repeats,
        "temperature": temperature,
        "workers": workers,
        "base_url": os.environ.get("LLM_BASE_URL"),
        "code_hashes": code_hashes,
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# ==========================================
# VALIDATE BASELINE RESULTS
# ==========================================

def valid_baseline(results, cases, repeats):
    """Check that baseline results are complete and error-free."""

    if not isinstance(results, dict):
        return False

    expected_ids = {
        case["id"] for case in cases
    }

    if set(results.keys()) != expected_ids:
        return False

    for case in cases:

        case_id = case["id"]
        result = results[case_id]

        if not isinstance(result, dict):
            return False

        runs = result.get("runs", [])

        if not isinstance(runs, list):
            return False

        if len(runs) != repeats:
            return False

        pass_rate = result.get("pass_rate")

        if not isinstance(pass_rate, (int, float)):
            return False

        if not 0 <= pass_rate <= 1:
            return False

        for run in runs:

            if not isinstance(run, dict):
                return False

            required_fields = (
                "text",
                "passed",
                "checks",
                "error",
                "latency",
                "cost",
            )

            if not all(
                field in run
                for field in required_fields
            ):
                return False

            if run["error"] is not None:
                return False

            if not isinstance(run["passed"], bool):
                return False

        expected_rate = (
            sum(run["passed"] for run in runs)
            / len(runs)
        )

        if abs(pass_rate - expected_rate) > 1e-9:
            return False

    return True


# ==========================================
# LOAD BASELINE CACHE
# ==========================================

def load_cached_baseline(cache_key, cases, repeats):
    """Load a previously evaluated baseline."""

    cache_path = CACHE_DIR / f"{cache_key}.json"

    if not cache_path.exists():
        return None

    try:

        payload = json.loads(
            cache_path.read_text(encoding="utf-8")
        )

        if payload.get("cache_key") != cache_key:
            return None

        results = payload.get("results")

        if not valid_baseline(
            results,
            cases,
            repeats
        ):
            print("Invalid baseline cache. Rebuilding...")
            return None

        return results

    except (OSError, ValueError, AttributeError, TypeError):
        print("Could not read baseline cache. Rebuilding...")
        return None


# ==========================================
# SAVE BASELINE CACHE
# ==========================================

def save_cached_baseline(
    cache_key,
    results,
    cases,
    repeats
):
    """Save only complete, error-free baseline evaluations."""

    if not valid_baseline(
        results,
        cases,
        repeats
    ):
        print("Baseline contains errors. Cache not saved.")
        return False

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = CACHE_DIR / f"{cache_key}.json"
    temporary_path = cache_path.with_suffix(".json.tmp")

    payload = {
        "cache_key": cache_key,
        "results": results,
    }

    temporary_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(cache_path)

    print(f"Baseline cache saved: {cache_path}")

    return True


# ==========================================
# MAIN CLI
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PromptGate: Automated LLM "
            "prompt regression testing"
        )
    )

    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the original prompt"
    )

    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to the modified prompt"
    )

    parser.add_argument(
        "--cases",
        default="testset/cases.jsonl"
    )

    parser.add_argument(
        "--config",
        default="promptgate.yaml"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases for testing"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override the number of repeated runs"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable baseline caching"
    )

    args = parser.parse_args()

    # ======================================
    # LOAD CONFIGURATION
    # ======================================

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    baseline_prompt = Path(args.baseline).read_text(
        encoding="utf-8"
    )

    candidate_prompt = Path(args.candidate).read_text(
        encoding="utf-8"
    )

    cases = load_cases(args.cases)

    if args.limit is not None:

        if args.limit < 1:
            parser.error("--limit must be at least 1")

        cases = cases[:args.limit]

    if not cases:
        parser.error("No evaluation cases found")

    case_ids = [
        case["id"] for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        parser.error("Evaluation case IDs must be unique")

    repeats = (
        args.repeats
        if args.repeats is not None
        else cfg.get("repeats", 3)
    )

    if repeats < 1:
        parser.error("--repeats must be at least 1")

    model = cfg["model"]
    temperature = cfg.get("temperature", 0.0)
    workers = cfg.get("workers", 2)

    # ======================================
    # CHECK BASELINE CACHE
    # ======================================

    base = None
    cache_key = None

    if not args.no_cache:

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
        calls_per_suite
        if cache_hit
        else 0
    )

    # ======================================
    # DISPLAY EVALUATION DETAILS
    # ======================================

    print(f"Model: {model}")
    print(f"Cases: {len(cases)}")
    print(f"Repeats: {repeats}")

    if args.no_cache:
        print("Baseline cache: DISABLED")
    elif cache_hit:
        print("Baseline cache: HIT")
    else:
        print("Baseline cache: MISS")

    print(f"Planned API calls: {planned_calls}")
    print(f"API calls saved by cache: {saved_calls}")

    # ======================================
    # EVALUATE BASELINE
    # ======================================

    if cache_hit:

        print("\nUsing cached baseline evaluation...")

    else:

        print("\nRunning baseline prompt...")

        base = run_suite(
            baseline_prompt,
            cases,
            model,
            repeats=repeats,
            temperature=temperature,
            workers=workers,
        )

        if not args.no_cache:

            try:
                save_cached_baseline(
                    cache_key,
                    base,
                    cases,
                    repeats,
                )

            except OSError as error:
                print(f"Could not save cache: {error}")

    # ======================================
    # EVALUATE CANDIDATE
    # ======================================

    print("Running candidate prompt...")

    cand = run_suite(
        candidate_prompt,
        cases,
        model,
        repeats=repeats,
        temperature=temperature,
        workers=workers,
    )

    # ======================================
    # COMPARE RESULTS
    # ======================================

    result = compare(
        base,
        cand,
        cases,
        cfg,
    )

    # ======================================
    # GENERATE REPORTS
    # ======================================

    Path("results").mkdir(exist_ok=True)

    with open(
        "results/comparison.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "comparison": result,
                "baseline_runs": base,
                "candidate_runs": cand,
                "cache": {
                    "enabled": not args.no_cache,
                    "baseline_cache_hit": cache_hit,
                    "cache_key": cache_key,
                    "planned_api_calls": planned_calls,
                    "api_calls_saved": saved_calls,
                },
            },
            file,
            indent=2,
        )

    write_html(
        "results/report.html",
        result,
        cases,
        base,
        cand,
    )

    write_markdown(
        "results/report.md",
        result,
        cases,
        base,
        cand,
    )

    # ======================================
    # DISPLAY FINAL VERDICT
    # ======================================

    print("\n" + "=" * 45)
    print(f"PROMPTGATE VERDICT: {result['verdict']}")
    print("=" * 45)

    print(f"Baseline score: {result['base']['score']:.1%}")
    print(f"Candidate score: {result['cand']['score']:.1%}")

    for reason in result["reasons"]:
        print(f"- {reason}")

    print(f"\nBaseline cache used: {cache_hit}")
    print(f"API calls saved: {saved_calls}")

    print("\nReports saved in results/")
    print("HTML: results/report.html")
    print("Markdown: results/report.md")

    if result["verdict"] == "PASS":
        return 0

    if result["verdict"] == "FAIL":
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
