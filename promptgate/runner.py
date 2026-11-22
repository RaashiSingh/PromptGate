
import json
from concurrent.futures import ThreadPoolExecutor

from .llm import complete
from .scoring import score_output


def load_cases(path):
    """Load evaluation cases from a JSONL file."""
    with open(path, encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def run_suite(
    prompt,
    cases,
    model,
    repeats=3,
    temperature=0.0,
    workers=2
):
    """
    Run every test case multiple times using the LLM.
    Score each response and calculate per-case pass rates.
    """

    jobs = [
        case
        for case in cases
        for _ in range(repeats)
    ]

    def work(case):
        output = complete(
            prompt,
            case["input"],
            model,
            temperature
        )

        scored = score_output(output["text"], case)

        return case["id"], {
            **output,
            **scored
        }

    results = {
        case["id"]: {"runs": []}
        for case in cases
    }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for case_id, run in executor.map(work, jobs):
            results[case_id]["runs"].append(run)

    for result in results.values():
        runs = result["runs"]

        result["pass_rate"] = (
            sum(run["passed"] for run in runs)
            / len(runs)
        )

    return results


def summarize(results, cases):
    """Calculate overall and group-level evaluation metrics."""

    groups = {}

    for case in cases:
        group = case["group"]
        case_id = case["id"]

        groups.setdefault(group, []).append(
            results[case_id]["pass_rate"]
        )

    all_runs = [
        run
        for result in results.values()
        for run in result["runs"]
    ]

    return {
        "score": (
            sum(
                result["pass_rate"]
                for result in results.values()
            ) / len(results)
        ),

        "groups": {
            group: sum(values) / len(values)
            for group, values in groups.items()
        },

        "cost": sum(
            run["cost"] for run in all_runs
        ),

        "latency": (
            sum(run["latency"] for run in all_runs)
            / len(all_runs)
        ),

        "input_tokens": sum(
            run.get("input_tokens", 0)
            for run in all_runs
        ),

        "output_tokens": sum(
            run.get("output_tokens", 0)
            for run in all_runs
        ),

        "errors": sum(
            1 for run in all_runs if run["error"]
        ),

        "calls": len(all_runs),
    }
