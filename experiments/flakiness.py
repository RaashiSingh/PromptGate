
import json
import statistics
from pathlib import Path

import yaml

from promptgate.runner import load_cases, run_suite, summarize


# Experiment configuration
TRIALS = 3
CASES_PER_TRIAL = 5
REPEATS_PER_CASE = 2

CHECKPOINT = Path("results/flakiness_checkpoint.json")
OUTPUT = Path("results/flakiness.json")


def save_json(path, data):
    """Save experiment data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main():

    # Load project configuration
    with open("promptgate.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    # Load original prompt
    prompt = Path("prompts/prompt.txt").read_text(
        encoding="utf-8"
    )

    # Load evaluation cases
    cases = load_cases("testset/cases.jsonl")[:CASES_PER_TRIAL]

    # Resume from checkpoint if available
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as file:
            checkpoint = json.load(file)

        scores = checkpoint.get("scores", [])

        expected = {
            "model": cfg["model"],
            "prompt": prompt,
            "case_ids": [case["id"] for case in cases],
            "repeats": REPEATS_PER_CASE,
            "temperature": cfg.get("temperature", 0.0),
        }

        if checkpoint.get("configuration") != expected:
            print("Checkpoint configuration does not match.")
            print("Move or delete the checkpoint before starting.")
            return

        print(f"Resuming from {len(scores)} completed trial(s).")

    else:
        scores = []

    configuration = {
        "model": cfg["model"],
        "prompt": prompt,
        "case_ids": [case["id"] for case in cases],
        "repeats": REPEATS_PER_CASE,
        "temperature": cfg.get("temperature", 0.0),
    }

    # Run remaining trials
    for trial in range(len(scores), TRIALS):

        print(f"\nRunning trial {trial + 1}/{TRIALS}")

        results = run_suite(
            prompt,
            cases,
            cfg["model"],
            repeats=REPEATS_PER_CASE,
            temperature=cfg.get("temperature", 0.0),
            workers=cfg.get("workers", 2),
        )

        summary = summarize(results, cases)

        # Detect and display API errors
        if summary["errors"]:

            print(
                f"\nDetected {summary['errors']} "
                "API error(s)."
            )

            for case_id, case_result in results.items():
                for run in case_result["runs"]:

                    if run.get("error"):
                        print(f"\nCase: {case_id}")
                        print(f"Error: {run['error']}")

            print("\nExperiment stopped.")
            print("Completed trials remain saved in checkpoint.")
            return

        # Record successful trial
        score = summary["score"]
        scores.append(score)

        print(f"Score: {score:.1%}")

        # Save checkpoint after every completed trial
        save_json(
            CHECKPOINT,
            {
                "configuration": configuration,
                "scores": scores,
                "completed_trials": len(scores),
            },
        )

        print("Checkpoint saved.")

    if not scores:
        print("No successful trials available.")
        return

    # Calculate statistics
    mean_score = statistics.mean(scores)
    score_std = statistics.pstdev(scores)
    score_range = max(scores) - min(scores)

    output = {
        "trials": len(scores),
        "cases_per_trial": len(cases),
        "repeats_per_case": REPEATS_PER_CASE,
        "scores": scores,
        "mean_score": mean_score,
        "score_std": score_std,
        "score_range": score_range,
    }

    # Save final analysis
    save_json(OUTPUT, output)

    print("\n" + "=" * 45)
    print("PROMPTGATE FLAKINESS ANALYSIS")
    print("=" * 45)

    print(f"Completed trials: {len(scores)}")
    print(f"Mean accuracy: {mean_score:.1%}")
    print(f"Standard deviation: {score_std:.1%}")
    print(f"Observed score range: {score_range:.1%}")

    print("\nResults saved to results/flakiness.json")


if __name__ == "__main__":
    main()
