
import json
from copy import deepcopy

import pytest

from promptgate import cli


# ==========================================
# TEST CONFIGURATION
# ==========================================

CASES = [
    {
        "id": "t01",
        "group": "critical",
        "input": "Customer was charged twice.",
        "priority": "high",
        "category": "billing",
    },
    {
        "id": "t02",
        "group": "standard",
        "input": "How do I update my profile?",
        "priority": "low",
        "category": "other",
    },
]

REPEATS = 2

PARAMETERS = {
    "prompt": "Classify the customer support ticket.",
    "cases": CASES,
    "model": "openai/gpt-oss-20b",
    "repeats": REPEATS,
    "temperature": 0.0,
    "workers": 2,
}


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """
    Use a temporary cache directory.
    Never modify actual benchmark cache files.
    """

    monkeypatch.setattr(
        cli,
        "CACHE_DIR",
        tmp_path / "baselines",
    )

    monkeypatch.setenv(
        "LLM_BASE_URL",
        "https://example.com/v1",
    )

    return cli.CACHE_DIR


@pytest.fixture
def valid_results():

    results = {}

    for case in CASES:

        runs = []

        for _ in range(REPEATS):

            runs.append({
                "text": json.dumps({
                    "priority": case["priority"],
                    "category": case["category"],
                    "summary": "Mock customer issue.",
                }),
                "passed": True,
                "checks": {
                    "valid_json": True,
                    "priority": True,
                    "category": True,
                    "summary": True,
                },
                "error": None,
                "latency": 0.5,
                "cost": 0.0,
                "input_tokens": 100,
                "output_tokens": 50,
            })

        results[case["id"]] = {
            "runs": runs,
            "pass_rate": 1.0,
        }

    return results


def make_key(**changes):

    parameters = deepcopy(PARAMETERS)
    parameters.update(changes)

    return cli.generate_cache_key(**parameters)


# ==========================================
# CACHE KEY TESTS
# ==========================================

def test_same_configuration_same_key():

    assert make_key() == make_key()


def test_cache_key_is_sha256():

    key = make_key()

    assert len(key) == 64
    assert all(
        char in "0123456789abcdef"
        for char in key
    )


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("prompt", "A different system prompt."),
        ("model", "different-model"),
        ("temperature", 0.7),
        ("repeats", 3),
        ("workers", 4),
    ],
)
def test_configuration_change_invalidates_cache(
    field, new_value
):

    original_key = make_key()

    modified_key = make_key(
        **{field: new_value}
    )

    assert original_key != modified_key


def test_changed_test_case_invalidates_cache():

    modified_cases = deepcopy(CASES)

    modified_cases[0]["input"] = (
        "A different customer complaint."
    )

    assert make_key() != make_key(
        cases=modified_cases
    )


def test_changed_expected_label_invalidates_cache():

    modified_cases = deepcopy(CASES)
    modified_cases[0]["priority"] = "medium"

    assert make_key() != make_key(
        cases=modified_cases
    )


def test_changed_api_endpoint_invalidates_cache(
    monkeypatch
):

    original_key = make_key()

    monkeypatch.setenv(
        "LLM_BASE_URL",
        "https://another-provider.example/v1",
    )

    assert original_key != make_key()


# ==========================================
# BASELINE VALIDATION TESTS
# ==========================================

def test_valid_baseline_accepted(valid_results):

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is True


def test_missing_case_rejected(valid_results):

    del valid_results["t01"]

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is False


def test_unexpected_case_rejected(valid_results):

    valid_results["extra"] = deepcopy(
        valid_results["t01"]
    )

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is False


def test_incomplete_runs_rejected(valid_results):

    valid_results["t01"]["runs"].pop()

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is False


def test_api_error_rejected(valid_results):

    valid_results["t01"]["runs"][0]["error"] = (
        "API rate limit exceeded"
    )

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is False


def test_missing_required_field_rejected(
    valid_results
):

    del valid_results["t01"]["runs"][0]["text"]

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is False


@pytest.mark.parametrize("rate", [-0.1, 1.1, "100%"])
def test_invalid_pass_rate_rejected(
    valid_results, rate
):

    valid_results["t01"]["pass_rate"] = rate

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is False


def test_inconsistent_pass_rate_rejected(
    valid_results
):

    valid_results["t01"]["pass_rate"] = 0.5

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is False


def test_non_boolean_passed_rejected(
    valid_results
):

    valid_results["t01"]["runs"][0]["passed"] = (
        "True"
    )

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is False


@pytest.mark.parametrize("invalid_results", [
    None,
    [],
    "invalid",
    42,
])
def test_invalid_result_structure_rejected(
    invalid_results
):

    assert cli.valid_baseline(
        invalid_results,
        CASES,
        REPEATS,
    ) is False


def test_valid_failed_assertion_can_be_cached(
    valid_results
):

    valid_results["t01"]["runs"][0]["passed"] = False

    valid_results["t01"]["runs"][0]["checks"][
        "priority"
    ] = False

    valid_results["t01"]["pass_rate"] = 0.5

    assert cli.valid_baseline(
        valid_results,
        CASES,
        REPEATS,
    ) is True


# ==========================================
# CACHE FILE TESTS
# ==========================================

def test_missing_cache_returns_none():

    result = cli.load_cached_baseline(
        make_key(),
        CASES,
        REPEATS,
    )

    assert result is None


def test_save_and_load_cache(valid_results):

    key = make_key()

    saved = cli.save_cached_baseline(
        key,
        valid_results,
        CASES,
        REPEATS,
    )

    loaded = cli.load_cached_baseline(
        key,
        CASES,
        REPEATS,
    )

    assert saved is True
    assert loaded == valid_results


def test_cache_file_created(
    valid_results, isolated_cache
):

    key = make_key()

    cli.save_cached_baseline(
        key,
        valid_results,
        CASES,
        REPEATS,
    )

    assert (
        isolated_cache / f"{key}.json"
    ).exists()


def test_corrupted_cache_rejected(
    isolated_cache
):

    key = make_key()

    isolated_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = isolated_cache / f"{key}.json"

    cache_file.write_text(
        "{invalid broken JSON",
        encoding="utf-8",
    )

    assert cli.load_cached_baseline(
        key,
        CASES,
        REPEATS,
    ) is None


def test_wrong_cache_key_rejected(
    valid_results, isolated_cache
):

    key = make_key()

    isolated_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "cache_key": "incorrect-cache-key",
        "results": valid_results,
    }

    cache_file = isolated_cache / f"{key}.json"

    cache_file.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert cli.load_cached_baseline(
        key,
        CASES,
        REPEATS,
    ) is None


def test_invalid_results_not_saved(
    valid_results, isolated_cache
):

    key = make_key()

    valid_results["t01"]["runs"][0]["error"] = (
        "API request failed"
    )

    saved = cli.save_cached_baseline(
        key,
        valid_results,
        CASES,
        REPEATS,
    )

    assert saved is False

    assert not (
        isolated_cache / f"{key}.json"
    ).exists()


def test_invalid_cached_payload_rejected(
    valid_results, isolated_cache
):

    key = make_key()

    isolated_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    del valid_results["t01"]

    payload = {
        "cache_key": key,
        "results": valid_results,
    }

    cache_file = isolated_cache / f"{key}.json"

    cache_file.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert cli.load_cached_baseline(
        key,
        CASES,
        REPEATS,
    ) is None


def test_no_temporary_file_left(
    valid_results, isolated_cache
):

    key = make_key()

    cli.save_cached_baseline(
        key,
        valid_results,
        CASES,
        REPEATS,
    )

    assert not (
        isolated_cache / f"{key}.json.tmp"
    ).exists()


def test_modified_prompt_cannot_reuse_old_cache(
    valid_results
):

    original_key = make_key()

    cli.save_cached_baseline(
        original_key,
        valid_results,
        CASES,
        REPEATS,
    )

    modified_key = make_key(
        prompt="Completely different prompt."
    )

    assert cli.load_cached_baseline(
        modified_key,
        CASES,
        REPEATS,
    ) is None