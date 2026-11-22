
import json
from threading import Barrier, Lock
from unittest.mock import Mock

import pytest

from promptgate import runner


# ==========================================
# TEST CONFIGURATION
# ==========================================

PROMPT = "Classify customer support tickets."
MODEL = "openai/gpt-oss-20b"

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


def make_response(passed=True, error=None):

    return {
        "text": json.dumps({
            "mock_passed": passed
        }),
        "latency": 0.5,
        "cost": 0.001,
        "input_tokens": 100,
        "output_tokens": 50,
        "error": error,
    }


def mock_score(text, case):

    try:
        passed = json.loads(text)["mock_passed"]
    except (ValueError, KeyError, TypeError):
        passed = False

    return {
        "passed": passed,
        "checks": {
            "valid_json": passed,
            "priority": passed,
            "category": passed,
            "summary": passed,
        },
    }


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mocked_llm(monkeypatch):

    mocked_complete = Mock(
        side_effect=lambda *args, **kwargs:
        make_response()
    )

    monkeypatch.setattr(
        runner,
        "complete",
        mocked_complete,
    )

    monkeypatch.setattr(
        runner,
        "score_output",
        mock_score,
    )

    return mocked_complete


@pytest.fixture
def sample_results():

    return {
        "t01": {
            "runs": [
                {
                    **make_response(True),
                    "passed": True,
                },
                {
                    **make_response(False),
                    "passed": False,
                },
            ],
            "pass_rate": 0.5,
        },
        "t02": {
            "runs": [
                {
                    **make_response(True),
                    "passed": True,
                },
                {
                    **make_response(True),
                    "passed": True,
                },
            ],
            "pass_rate": 1.0,
        },
    }


# ==========================================
# JSONL LOADING TESTS
# ==========================================

def test_load_single_case(tmp_path):

    path = tmp_path / "cases.jsonl"

    path.write_text(
        json.dumps(CASES[0]),
        encoding="utf-8",
    )

    assert runner.load_cases(path) == [
        CASES[0]
    ]


def test_load_multiple_cases(tmp_path):

    path = tmp_path / "cases.jsonl"

    path.write_text(
        "\n".join(
            json.dumps(case)
            for case in CASES
        ),
        encoding="utf-8",
    )

    assert runner.load_cases(path) == CASES


def test_load_cases_ignores_blank_lines(tmp_path):

    path = tmp_path / "cases.jsonl"

    path.write_text(
        "\n\n"
        + json.dumps(CASES[0])
        + "\n   \n"
        + json.dumps(CASES[1])
        + "\n\n",
        encoding="utf-8",
    )

    assert runner.load_cases(path) == CASES


def test_load_empty_file(tmp_path):

    path = tmp_path / "cases.jsonl"

    path.write_text(
        "",
        encoding="utf-8",
    )

    assert runner.load_cases(path) == []


def test_missing_file_raises_error(tmp_path):

    with pytest.raises(FileNotFoundError):

        runner.load_cases(
            tmp_path / "missing.jsonl"
        )


def test_invalid_json_raises_error(tmp_path):

    path = tmp_path / "cases.jsonl"

    path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):

        runner.load_cases(path)


# ==========================================
# RUNNER EXECUTION TESTS
# ==========================================

def test_single_case_single_run(mocked_llm):

    results = runner.run_suite(
        PROMPT,
        [CASES[0]],
        MODEL,
        repeats=1,
        workers=1,
    )

    assert len(results) == 1
    assert len(results["t01"]["runs"]) == 1
    assert results["t01"]["pass_rate"] == 1.0

    mocked_llm.assert_called_once()


def test_multiple_cases(mocked_llm):

    results = runner.run_suite(
        PROMPT,
        CASES,
        MODEL,
        repeats=1,
        workers=2,
    )

    assert set(results) == {"t01", "t02"}
    assert mocked_llm.call_count == 2


@pytest.mark.parametrize(
    "repeats",
    [1, 2, 3, 5],
)
def test_repeat_count(mocked_llm, repeats):

    results = runner.run_suite(
        PROMPT,
        CASES,
        MODEL,
        repeats=repeats,
        workers=2,
    )

    for case in CASES:

        assert len(
            results[case["id"]]["runs"]
        ) == repeats

    assert mocked_llm.call_count == (
        len(CASES) * repeats
    )


def test_correct_api_arguments(mocked_llm):

    runner.run_suite(
        PROMPT,
        [CASES[0]],
        MODEL,
        repeats=1,
        temperature=0.3,
        workers=1,
    )

    mocked_llm.assert_called_once_with(
        PROMPT,
        CASES[0]["input"],
        MODEL,
        0.3,
    )


def test_all_passed(mocked_llm):

    results = runner.run_suite(
        PROMPT,
        CASES,
        MODEL,
        repeats=3,
    )

    assert all(
        result["pass_rate"] == 1.0
        for result in results.values()
    )


def test_all_failed(mocked_llm):

    mocked_llm.side_effect = None
    mocked_llm.return_value = make_response(
        False
    )

    results = runner.run_suite(
        PROMPT,
        CASES,
        MODEL,
        repeats=3,
    )

    assert all(
        result["pass_rate"] == 0.0
        for result in results.values()
    )


def test_partial_pass_rate(mocked_llm):

    mocked_llm.side_effect = [
        make_response(True),
        make_response(False),
        make_response(True),
        make_response(False),
    ]

    results = runner.run_suite(
        PROMPT,
        [CASES[0]],
        MODEL,
        repeats=4,
        workers=1,
    )

    assert results["t01"]["pass_rate"] == (
        pytest.approx(0.5)
    )


def test_response_metadata_preserved(mocked_llm):

    results = runner.run_suite(
        PROMPT,
        [CASES[0]],
        MODEL,
        repeats=1,
    )

    run = results["t01"]["runs"][0]

    assert run["latency"] == 0.5
    assert run["cost"] == 0.001
    assert run["input_tokens"] == 100
    assert run["output_tokens"] == 50
    assert run["error"] is None
    assert run["passed"] is True
    assert "checks" in run
    assert "text" in run


def test_empty_case_list(mocked_llm):

    results = runner.run_suite(
        PROMPT,
        [],
        MODEL,
        repeats=1,
    )

    assert results == {}
    mocked_llm.assert_not_called()


# ==========================================
# CONCURRENT EXECUTION TESTS
# ==========================================

@pytest.mark.parametrize(
    "workers",
    [1, 2, 4],
)
def test_worker_configurations(
    mocked_llm, workers
):

    results = runner.run_suite(
        PROMPT,
        CASES,
        MODEL,
        repeats=5,
        workers=workers,
    )

    assert mocked_llm.call_count == 10

    assert len(
        results["t01"]["runs"]
    ) == 5

    assert len(
        results["t02"]["runs"]
    ) == 5


def test_concurrent_case_mapping(
    mocked_llm
):

    def response_by_input(
        prompt,
        user_input,
        model,
        temperature,
    ):

        passed = (
            user_input == CASES[0]["input"]
        )

        return make_response(passed)

    mocked_llm.side_effect = response_by_input

    results = runner.run_suite(
        PROMPT,
        CASES,
        MODEL,
        repeats=10,
        workers=4,
    )

    assert results["t01"]["pass_rate"] == 1.0
    assert results["t02"]["pass_rate"] == 0.0
    assert mocked_llm.call_count == 20


def test_workers_execute_concurrently(
    mocked_llm
):

    barrier = Barrier(2)
    lock = Lock()

    active = 0
    peak_active = 0

    def concurrent_response(*args):

        nonlocal active, peak_active

        with lock:

            active += 1

            peak_active = max(
                peak_active,
                active,
            )

        try:

            barrier.wait(timeout=5)

            return make_response(True)

        finally:

            with lock:
                active -= 1

    mocked_llm.side_effect = (
        concurrent_response
    )

    results = runner.run_suite(
        PROMPT,
        CASES,
        MODEL,
        repeats=1,
        workers=2,
    )

    assert peak_active == 2

    assert results["t01"]["pass_rate"] == 1.0
    assert results["t02"]["pass_rate"] == 1.0


def test_invalid_worker_count(mocked_llm):

    with pytest.raises(ValueError):

        runner.run_suite(
            PROMPT,
            CASES,
            MODEL,
            repeats=1,
            workers=0,
        )


# ==========================================
# ERROR HANDLING TESTS
# ==========================================

def test_api_error_preserved(mocked_llm):

    response = make_response(
        False,
        error="API rate limit exceeded",
    )

    response["text"] = ""
    response["latency"] = 0.0
    response["cost"] = 0.0
    response["input_tokens"] = 0
    response["output_tokens"] = 0

    mocked_llm.side_effect = None
    mocked_llm.return_value = response

    results = runner.run_suite(
        PROMPT,
        [CASES[0]],
        MODEL,
        repeats=1,
    )

    run = results["t01"]["runs"][0]

    assert run["error"] == (
        "API rate limit exceeded"
    )

    assert run["passed"] is False
    assert results["t01"]["pass_rate"] == 0.0


def test_mixed_success_and_api_error(
    mocked_llm
):

    failed_response = make_response(
        False,
        error="API request failed",
    )

    failed_response["text"] = ""

    mocked_llm.side_effect = [
        make_response(True),
        failed_response,
    ]

    results = runner.run_suite(
        PROMPT,
        [CASES[0]],
        MODEL,
        repeats=2,
        workers=1,
    )

    assert results["t01"]["pass_rate"] == 0.5

    assert results["t01"]["runs"][1][
        "error"
    ] == "API request failed"


def test_unexpected_exception_propagates(
    mocked_llm
):

    mocked_llm.side_effect = RuntimeError(
        "Unexpected execution failure"
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected execution failure",
    ):

        runner.run_suite(
            PROMPT,
            [CASES[0]],
            MODEL,
            repeats=1,
            workers=1,
        )


# ==========================================
# SUMMARY METRIC TESTS
# ==========================================

def test_overall_score(sample_results):

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["score"] == (
        pytest.approx(0.75)
    )


def test_group_scores(sample_results):

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["groups"] == {
        "critical": 0.5,
        "standard": 1.0,
    }


def test_total_cost(sample_results):

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["cost"] == (
        pytest.approx(0.004)
    )


def test_average_latency(sample_results):

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["latency"] == (
        pytest.approx(0.5)
    )


def test_total_input_tokens(sample_results):

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["input_tokens"] == 400


def test_total_output_tokens(sample_results):

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["output_tokens"] == 200


def test_total_api_calls(sample_results):

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["calls"] == 4


def test_zero_execution_errors(sample_results):

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["errors"] == 0


def test_execution_error_count(sample_results):

    sample_results["t01"]["runs"][0][
        "error"
    ] = "Timeout"

    sample_results["t02"]["runs"][1][
        "error"
    ] = "Rate limit"

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["errors"] == 2


def test_missing_token_metadata(
    sample_results
):

    for result in sample_results.values():

        for run in result["runs"]:

            run.pop("input_tokens", None)
            run.pop("output_tokens", None)

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["input_tokens"] == 0
    assert summary["output_tokens"] == 0


def test_group_average_multiple_cases(
    sample_results
):

    cases = [
        {**case, "group": "critical"}
        for case in CASES
    ]

    summary = runner.summarize(
        sample_results,
        cases,
    )

    assert summary["groups"]["critical"] == (
        pytest.approx(0.75)
    )


def test_variable_latency(sample_results):

    sample_results["t01"]["runs"][0][
        "latency"
    ] = 1.0

    summary = runner.summarize(
        sample_results,
        CASES,
    )

    assert summary["latency"] == (
        pytest.approx(0.625)
    )
