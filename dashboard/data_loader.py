
import json
from pathlib import Path

import pandas as pd


# =============================================
# PROJECT PATHS
# =============================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results"

EXPERIMENTS_DIR = ROOT_DIR / "experiments"

MANIFEST_PATH = (
    EXPERIMENTS_DIR / "mutations_manifest.json"
)

CASES_PATH = ROOT_DIR / "testset" / "cases.jsonl"


# =============================================
# JSON UTILITIES
# =============================================

def read_json(path):
    """Read a JSON file."""

    with open(path, encoding="utf-8-sig") as file:
        return json.load(file)


# =============================================
# DISCOVER BENCHMARK VERSIONS
# =============================================

def get_available_benchmarks():
    """Find all saved benchmark versions."""

    benchmarks = {}

    for folder in RESULTS_DIR.glob("benchmark_v*"):

        if not folder.is_dir():
            continue

        summary_file = folder / "benchmark.json"

        if summary_file.exists():

            version = folder.name.replace(
                "benchmark_", ""
            ).upper()

            benchmarks[version] = folder

    return dict(sorted(benchmarks.items()))


# =============================================
# LOAD MUTATION METADATA
# =============================================

def load_manifest():
    """Load mutation names, categories and paths."""

    manifest = read_json(MANIFEST_PATH)

    return {
        item["id"]: item
        for item in manifest
    }


# =============================================
# LOAD TEST CASES
# =============================================

def load_test_cases():
    """Load original evaluation test cases."""

    with open(
        CASES_PATH,
        encoding="utf-8-sig"
    ) as file:

        cases = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    return {
        case["id"]: case
        for case in cases
    }


# =============================================
# LOAD INDIVIDUAL MUTATION REPORT
# =============================================

def load_mutation_report(version, mutation_id):
    """Read one saved comparison report."""

    benchmarks = get_available_benchmarks()

    if version not in benchmarks:
        raise ValueError(
            f"Unknown benchmark: {version}"
        )

    report_path = (
        benchmarks[version]
        / "mutations"
        / mutation_id
        / "comparison.json"
    )

    if not report_path.exists():
        return None

    return read_json(report_path)


# =============================================
# LOAD COMPLETE BENCHMARK
# =============================================

def load_benchmark(version):
    """Convert benchmark reports into a DataFrame."""

    benchmarks = get_available_benchmarks()

    if version not in benchmarks:
        raise ValueError(
            f"Unknown benchmark: {version}"
        )

    benchmark_path = benchmarks[version]

    summary = read_json(
        benchmark_path / "benchmark.json"
    )

    manifest = load_manifest()

    rows = []

    for item in summary:

        mutation_id = item["mutation_id"]

        metadata = manifest.get(
            mutation_id, {}
        )

        mutation_type = item["mutation_type"]

        report = load_mutation_report(
            version,
            mutation_id
        )

        if report is None:

            rows.append({
                "mutation_id": mutation_id,
                "name": metadata.get(
                    "name", mutation_id
                ),
                "type": mutation_type,
                "verdict": "ERROR",
                "expected_verdict": (
                    "FAIL"
                    if mutation_type == "regression"
                    else "PASS"
                ),
                "outcome": "Execution error",
                "duration_seconds": item.get(
                    "duration_seconds", 0
                ),
                "report_available": False,
            })

            continue

        comparison = report["comparison"]

        base = comparison["base"]
        cand = comparison["cand"]

        cache = report.get("cache", {})

        baseline_runs = report.get(
            "baseline_runs", {}
        )

        candidate_runs = report.get(
            "candidate_runs", {}
        )

        verdict = comparison["verdict"]

        expected_verdict = (
            "FAIL"
            if mutation_type == "regression"
            else "PASS"
        )

        if verdict == "ERROR":

            outcome = "Execution error"

        elif mutation_type == "regression":

            outcome = (
                "Detected regression"
                if verdict == "FAIL"
                else "Missed regression"
            )

        else:

            outcome = (
                "Accepted harmless change"
                if verdict == "PASS"
                else "False alarm"
            )

        baseline_calls = base.get("calls", 0)

        candidate_calls = cand.get("calls", 0)

        uncached_calls = (
            baseline_calls + candidate_calls
        )

        planned_calls = cache.get(
            "planned_api_calls",
            uncached_calls
        )

        saved_calls = cache.get(
            "api_calls_saved", 0
        )

        cache_hit = cache.get(
            "baseline_cache_hit", False
        )

        baseline_tokens = (
            base.get("input_tokens", 0)
            + base.get("output_tokens", 0)
        )

        candidate_tokens = (
            cand.get("input_tokens", 0)
            + cand.get("output_tokens", 0)
        )

        new_tokens = (
            candidate_tokens
            if cache_hit
            else baseline_tokens + candidate_tokens
        )

        rows.append({

            "mutation_id": mutation_id,

            "name": metadata.get(
                "name", mutation_id
            ),

            "type": mutation_type,

            "verdict": verdict,

            "expected_verdict": expected_verdict,

            "outcome": outcome,

            "baseline_score": base["score"],

            "candidate_score": cand["score"],

            "score_drop": (
                base["score"] - cand["score"]
            ),

            "cases": len(candidate_runs),

            "baseline_evaluations": baseline_calls,

            "candidate_evaluations": candidate_calls,

            "critical_flips": len(
                comparison.get("flips_down", [])
            ),

            "improvements": len(
                comparison.get("flips_up", [])
            ),

            "baseline_latency": base.get(
                "latency", 0
            ),

            "candidate_latency": cand.get(
                "latency", 0
            ),

            "baseline_tokens": baseline_tokens,

            "candidate_tokens": candidate_tokens,

            "new_tokens": new_tokens,

            "baseline_errors": base.get(
                "errors", 0
            ),

            "candidate_errors": cand.get(
                "errors", 0
            ),

            "cache_hit": cache_hit,

            "cache_enabled": cache.get(
                "enabled", False
            ),

            "uncached_calls": uncached_calls,

            "planned_calls": planned_calls,

            "saved_calls": saved_calls,

            "duration_seconds": item.get(
                "duration_seconds", 0
            ),

            "report_available": True,

        })

    df = pd.DataFrame(rows)

    if not df.empty:

        df = df.sort_values(
            "mutation_id"
        ).reset_index(drop=True)

    return df


# =============================================
# CALCULATE BENCHMARK METRICS
# =============================================

def calculate_metrics(df):
    """Calculate dashboard KPIs dynamically."""

    def count_outcome(name):

        return int(
            (df["outcome"] == name).sum()
        )

    def total(column):

        if column not in df.columns:
            return 0

        return int(
            df[column].fillna(0).sum()
        )

    detected = count_outcome(
        "Detected regression"
    )

    missed = count_outcome(
        "Missed regression"
    )

    accepted = count_outcome(
        "Accepted harmless change"
    )

    false_alarms = count_outcome(
        "False alarm"
    )

    errors = count_outcome(
        "Execution error"
    )

    completed_regressions = detected + missed

    completed_harmless = (
        accepted + false_alarms
    )

    detection_rate = (
        detected / completed_regressions
        if completed_regressions
        else None
    )

    false_alarm_rate = (
        false_alarms / completed_harmless
        if completed_harmless
        else None
    )

    uncached_calls = total("uncached_calls")

    planned_calls = total("planned_calls")

    saved_calls = total("saved_calls")

    reduction = (
        saved_calls / uncached_calls
        if uncached_calls
        else 0
    )

    if "cache_hit" in df.columns:

        cache_hits = int(
            df["cache_hit"].eq(True).sum()
        )

        cache_misses = int(
            (
                df["cache_enabled"].eq(True)
                & df["cache_hit"].eq(False)
            ).sum()
        )

    else:

        cache_hits = 0
        cache_misses = 0

    cache_checks = cache_hits + cache_misses

    cache_hit_rate = (
        cache_hits / cache_checks
        if cache_checks
        else None
    )

    baseline_accuracy = (
        float(df["baseline_score"].mean())
        if "baseline_score" in df.columns
        else None
    )

    return {

        "total_mutations": len(df),

        "completed_mutations": (
            len(df) - errors
        ),

        "detected_regressions": detected,

        "missed_regressions": missed,

        "accepted_harmless": accepted,

        "false_alarms": false_alarms,

        "execution_errors": errors,

        "detection_rate": detection_rate,

        "false_alarm_rate": false_alarm_rate,

        "baseline_accuracy": baseline_accuracy,

        "candidate_evaluations": total(
            "candidate_evaluations"
        ),

        "uncached_calls": uncached_calls,

        "planned_calls": planned_calls,

        "saved_calls": saved_calls,

        "call_reduction": reduction,

        "cache_hits": cache_hits,

        "cache_misses": cache_misses,

        "cache_hit_rate": cache_hit_rate,

        "new_tokens": total("new_tokens"),

    }


# =============================================
# BUILD REGRESSION HEATMAP DATA
# =============================================

def build_heatmap_data(version):
    """Create mutation-by-case comparison data."""

    df = load_benchmark(version)

    rows = []

    for mutation_id in df["mutation_id"]:

        report = load_mutation_report(
            version,
            mutation_id
        )

        if report is None:
            continue

        baseline = report.get(
            "baseline_runs", {}
        )

        candidate = report.get(
            "candidate_runs", {}
        )

        for case_id, base_result in baseline.items():

            if case_id not in candidate:
                continue

            cand_result = candidate[case_id]

            base_rate = base_result["pass_rate"]

            cand_rate = cand_result["pass_rate"]

            has_error = any(
                run.get("error")
                for run in (
                    base_result["runs"]
                    + cand_result["runs"]
                )
            )

            if has_error:

                state = "Execution error"

            elif cand_rate < base_rate:

                state = "Regression"

            elif cand_rate > base_rate:

                state = "Improvement"

            elif cand_rate == 1:

                state = "Stable pass"

            elif cand_rate == 0:

                state = "Existing failure"

            else:

                state = "Stable partial pass"

            rows.append({

                "mutation_id": mutation_id,

                "case_id": case_id,

                "baseline_rate": base_rate,

                "candidate_rate": cand_rate,

                "change": cand_rate - base_rate,

                "state": state,

            })

    return pd.DataFrame(rows)


# =============================================
# LOAD PROMPT TEXT
# =============================================

def load_prompt_pair(mutation_id):
    """Load current baseline and mutated prompt."""

    manifest = load_manifest()

    if mutation_id not in manifest:
        raise ValueError("Unknown mutation")

    baseline_path = (
        ROOT_DIR / "prompts" / "prompt.txt"
    )

    candidate_path = (
        EXPERIMENTS_DIR
        / "mutated_prompts"
        / f"{mutation_id}.txt"
    )

    baseline = baseline_path.read_text(
        encoding="utf-8"
    )

    candidate = candidate_path.read_text(
        encoding="utf-8"
    )

    return baseline, candidate
