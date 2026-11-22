
import json
import re


def extract_json(text):
    """Parse a JSON object, including responses with Markdown fences."""
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?|```$",
        "",
        text,
        flags=re.MULTILINE
    ).strip()

    candidates = [text]

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)

            if isinstance(obj, dict):
                return obj

        except (json.JSONDecodeError, ValueError):
            continue

    return None


def score_output(text, case, max_words=30):
    """
    Evaluate the AI response against expected labels.
    All checks must pass for the case to pass.
    """

    obj = extract_json(text)

    if obj is None:
        return {
            "passed": False,
            "checks": {
                "valid_json": False,
                "priority": False,
                "category": False,
                "summary": False,
            },
        }

    summary = obj.get("summary")

    checks = {
        "valid_json": True,

        "priority": (
            str(obj.get("priority", "")).strip().lower()
            == case["priority"]
        ),

        "category": (
            str(obj.get("category", "")).strip().lower()
            == case["category"]
        ),

        "summary": (
            isinstance(summary, str)
            and 0 < len(summary.split()) <= max_words
        ),
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
    }
