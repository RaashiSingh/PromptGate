
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = ROOT / "experiments" / "mutations_manifest.json"
OUTPUT = ROOT / "results" / "benchmark.json"

BASELINE = ROOT / "prompts" / "prompt.txt"
CASES = ROOT / "testset" / "cases.jsonl"
CONFIG = ROOT / "promptgate.yaml"

REPORTS_DIR = ROOT / "results" / "mutations"

REPORT_FILES = (
    "comparison.json",
    "report.html",
    "report.md",
)


# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================

# None = evaluate all mutations.
MAX_MUTATIONS = None

# Keep the current small test for now.
LIMIT = 10
REPEATS = 1

TIMEOUT_SECONDS = 300
COOLDOWN_SECONDS = 30

# Skip previously completed, matching evaluations.
RESUME = True

# Stop after an execution/API error.
STOP_ON_ERROR = True


# ==========================================
# FILE FINGERPRINTS
# ==========================================

def file_hash(path):
    """Return a SHA-256 hash of a file."""

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def evaluation_fingerprint(candidate):
    """
    Identify the exact inputs and settings
    used for a mutation evaluation.
    """

    files = (
        BASELINE,
        candidate,
        CASES,
        CONFIG,
        ROOT / "promptgate" / "llm.py",
        ROOT / "promptgate" / "runner.py",
        ROOT / "promptgate" / "scoring.py",
        ROOT / "promptgate" / "compare.py",
        ROOT / "promptgate" / "cli.py",
    )

    payload = {
        "limit": LIMIT,
        "repeats": REPEATS,
        "files": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in files
        },
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# ==========================================
# SAVE AND LOAD RESULTS
# ==========================================

def save_results(results):
    """Atomically save benchmark progress."""

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = OUTPUT.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    temporary.replace(OUTPUT)


def load_previous_results():
    """Load previous benchmark results if available."""

    if not RESUME or not OUTPUT.exists():
        return {}

    try:
        saved = json.loads(
            OUTPUT.read_text(encoding="utf-8")
        )

        if not isinstance(saved, list):
            print("Previous results are not a list.")
            return {}

        return {
            item["mutation_id"]: item
            for item in saved
            if isinstance(item, dict)
            and "mutation_id" in item
        }

    except (OSError, ValueError) as error:
        print(f"Could not load previous results: {error}")
        return {}


# ==========================================
# REPORT MANAGEMENT
# ==========================================

def report_directory(mutation_id):
    return REPORTS_DIR / mutation_id


def clear_shared_reports():
    """Prevent stale reports from being reused."""

    for filename in REPORT_FILES:
        path = ROOT / "results" / filename

        if path.exists():
            path.unlink()


def preserve_reports(mutation_id):
    """Save this mutation's reports separately."""

    destination = report_directory(mutation_id)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in REPORT_FILES:
        source = ROOT / "results" / filename

        if source.exists():
            shutil.copy2(
                source,
                destination / filename,
            )


def read_comparison(path):
    """Read and validate a comparison report."""

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        verdict = data["comparison"]["verdict"]

        if verdict not in ("PASS", "FAIL", "ERROR"):
            return None

        return data

    except (OSError, ValueError, KeyError, TypeError):
        return None


# ==========================================
# RESUME VALIDATION
# ==========================================

def can_resume(previous, fingerprint, mutation_id):
    """
    Reuse only successful completed evaluations
    with matching fingerprints and reports.
    """

    if previous is None:
        return False

    if previous.get("status") not in ("PASS", "FAIL"):
        return False

    if previous.get("fingerprint") != fingerprint:
        return False

    report_path = (
        report_directory(mutation_id)
        / "comparison.json"
    )

    report = read_comparison(report_path)

    if report is None:
        return False

    return (
        report["comparison"]["verdict"]
        == previous["status"]
    )


# ==========================================
# CLASSIFY CLI RESULT
# ==========================================

def classify_result(process):
    """Verify CLI return code and report verdict."""

    expected_codes = {
        "PASS": 0,
        "FAIL": 1,
        "ERROR": 2,
    }

    comparison_path = (
        ROOT / "results" / "comparison.json"
    )

    report = read_comparison(comparison_path)

    if report is None:
        return "ERROR"

    verdict = report["comparison"]["verdict"]

    if expected_codes[verdict] != process.returncode:
        return "ERROR"

    return verdict


# ==========================================
# RUN ONE MUTATION
# ==========================================

def run_mutation(mutation, candidate, fingerprint):

    mutation_id = mutation["id"]
    mutation_type = mutation.get("type", "unknown")

    command = [
        sys.executable,
        "-m",
        "promptgate.cli",
        "--baseline",
        str(BASELINE),
        "--candidate",
        str(candidate),
        "--cases",
        str(CASES),
        "--config",
        str(CONFIG),
        "--limit",
        str(LIMIT),
        "--repeats",
        str(REPEATS),
    ]

    clear_shared_reports()

    start_time = time.perf_counter()

    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
        )

        duration = round(
            time.perf_counter() - start_time,
            2,
        )

        status = classify_result(process)

        result = {
            "mutation_id": mutation_id,
            "mutation_type": mutation_type,
            "status": status,
            "fingerprint": fingerprint,
            "returncode": process.returncode,
            "duration_seconds": duration,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }

    except subprocess.TimeoutExpired:
        result = {
            "mutation_id": mutation_id,
            "mutation_type": mutation_type,
            "status": "ERROR",
            "fingerprint": fingerprint,
            "error": "Execution timed out",
            "duration_seconds": round(
                time.perf_counter() - start_time,
                2,
            ),
        }

    except Exception as error:
        result = {
            "mutation_id": mutation_id,
            "mutation_type": mutation_type,
            "status": "ERROR",
            "fingerprint": fingerprint,
            "error": str(error),
        }

    comparison_path = (
        ROOT / "results" / "comparison.json"
    )

    if comparison_path.exists():
        preserve_reports(mutation_id)

    return result


# ==========================================
# SUMMARY
# ==========================================

def print_summary(results, total_mutations):

    print("\n" + "=" * 45)
    print("PROMPTGATE MUTATION BENCHMARK")
    print("=" * 45)

    passed = sum(
        r["status"] == "PASS"
        for r in results
    )

    failed = sum(
        r["status"] == "FAIL"
        for r in results
    )

    errors = sum(
        r["status"] == "ERROR"
        for r in results
    )

    detected = sum(
        r["mutation_type"] == "regression"
        and r["status"] == "FAIL"
        for r in results
    )

    missed = sum(
        r["mutation_type"] == "regression"
        and r["status"] == "PASS"
        for r in results
    )

    false_alarms = sum(
        r["mutation_type"] == "harmless"
        and r["status"] == "FAIL"
        for r in results
    )

    completed_regressions = detected + missed

    completed_harmless = sum(
        r["mutation_type"] == "harmless"
        and r["status"] in ("PASS", "FAIL")
        for r in results
    )

    print(
        f"Mutations evaluated or reused: "
        f"{len(results)}/{total_mutations}"
    )

    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print(f"ERROR: {errors}")

    print(f"\nDetected regressions: {detected}")
    print(f"Missed regressions: {missed}")
    print(f"False alarms: {false_alarms}")

    if completed_regressions:
        print(
            "Detection rate (completed tests): "
            f"{detected / completed_regressions:.1%}"
        )
    else:
        print("Detection rate: N/A")

    if completed_harmless:
        print(
            "False-alarm rate (completed tests): "
            f"{false_alarms / completed_harmless:.1%}"
        )
    else:
        print("False-alarm rate: N/A")

    print(f"\nResults saved to: {OUTPUT}")


# ==========================================
# MAIN
# ==========================================

def main():

    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        return 1

    mutations = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    print(f"Loaded {len(mutations)} mutations")

    if MAX_MUTATIONS is not None:
        mutations = mutations[:MAX_MUTATIONS]

    previous_results = load_previous_results()

    if previous_results:
        print(
            f"Found {len(previous_results)} "
            "previous benchmark result(s)."
        )

    results = []

    for index, mutation in enumerate(mutations):

        mutation_id = mutation["id"]
        mutation_type = mutation.get("type", "unknown")

        print(
            f"\nMutation {index + 1}/"
            f"{len(mutations)}: {mutation_id}",
            flush=True,
        )

        candidate = (
            ROOT
            / "experiments"
            / "mutated_prompts"
            / f"{mutation_id}.txt"
        )

        if not candidate.exists():
            result = {
                "mutation_id": mutation_id,
                "mutation_type": mutation_type,
                "status": "ERROR",
                "error": f"Missing candidate: {candidate}",
            }

            results.append(result)
            save_results(results)

            print("Result: ERROR (missing candidate)")
            break

        fingerprint = evaluation_fingerprint(candidate)

        previous = previous_results.get(mutation_id)

        if can_resume(
            previous,
            fingerprint,
            mutation_id,
        ):
            print(
                f"Reusing previous result: "
                f"{previous['status']}"
            )

            results.append(previous)
            save_results(results)
            continue

        print("Running fresh evaluation...")

        result = run_mutation(
            mutation,
            candidate,
            fingerprint,
        )

        results.append(result)
        save_results(results)

        print(f"Result: {result['status']}")

        if result["status"] == "ERROR" and STOP_ON_ERROR:
            print(
                "\nBenchmark paused due to an error."
            )
            print(
                "Fix the error and rerun to resume."
            )
            break

        if index < len(mutations) - 1:
            print(
                f"Cooling down for "
                f"{COOLDOWN_SECONDS} seconds..."
            )
            time.sleep(COOLDOWN_SECONDS)

    print_summary(results, len(mutations))

    if any(r["status"] == "ERROR" for r in results):
        return 2

    if len(results) < len(mutations):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
