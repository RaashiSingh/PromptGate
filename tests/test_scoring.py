
import json
import pytest

from promptgate.scoring import extract_json, score_output


CASE = {
    "priority": "high",
    "category": "billing"
}

GOOD = {
    "priority": "high",
    "category": "billing",
    "summary": "Customer was charged twice."
}


# ==========================================
# JSON EXTRACTION TESTS
# ==========================================

def test_extract_valid_json():
    assert extract_json(json.dumps(GOOD)) == GOOD


def test_extract_markdown_json():
    text = "```json\n" + json.dumps(GOOD) + "\n```"
    assert extract_json(text) == GOOD


def test_extract_json_with_surrounding_text():
    text = "Here is the response:\n" + json.dumps(GOOD)
    assert extract_json(text) == GOOD


@pytest.mark.parametrize("text", [
    "",
    "Not a JSON response",
    '{"priority":',
    "[]",
    "null",
    "42"
])
def test_extract_invalid_json(text):
    assert extract_json(text) is None


# ==========================================
# RESPONSE SCORING TESTS
# ==========================================

def test_correct_output():
    result = score_output(json.dumps(GOOD), CASE)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_incorrect_output():
    bad = {**GOOD, "priority": "low"}

    result = score_output(json.dumps(bad), CASE)

    assert result["passed"] is False
    assert result["checks"]["priority"] is False


def test_incorrect_category():
    bad = {**GOOD, "category": "other"}

    result = score_output(json.dumps(bad), CASE)

    assert result["passed"] is False
    assert result["checks"]["category"] is False


@pytest.mark.parametrize("field", [
    "priority",
    "category",
    "summary"
])
def test_missing_required_field(field):
    bad = GOOD.copy()
    del bad[field]

    result = score_output(json.dumps(bad), CASE)

    assert result["passed"] is False
    assert result["checks"][field] is False


@pytest.mark.parametrize("summary", [
    "",
    "   ",
    None,
    123,
    ["Customer complaint"]
])
def test_invalid_summary(summary):
    bad = {**GOOD, "summary": summary}

    result = score_output(json.dumps(bad), CASE)

    assert result["passed"] is False
    assert result["checks"]["summary"] is False


def test_summary_at_word_limit():
    output = {
        **GOOD,
        "summary": " ".join(["word"] * 30)
    }

    result = score_output(json.dumps(output), CASE)

    assert result["passed"] is True


def test_summary_exceeds_word_limit():
    output = {
        **GOOD,
        "summary": " ".join(["word"] * 31)
    }

    result = score_output(json.dumps(output), CASE)

    assert result["passed"] is False
    assert result["checks"]["summary"] is False


def test_case_insensitive_labels():
    output = {
        **GOOD,
        "priority": " HIGH ",
        "category": " BILLING "
    }

    result = score_output(json.dumps(output), CASE)

    assert result["passed"] is True


def test_invalid_json_fails_all_checks():
    result = score_output("Invalid response", CASE)

    assert result["passed"] is False
    assert not any(result["checks"].values())


def test_custom_summary_word_limit():
    result = score_output(
        json.dumps(GOOD),
        CASE,
        max_words=2
    )

    assert result["passed"] is False
    assert result["checks"]["summary"] is False
