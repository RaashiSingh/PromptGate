
import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from promptgate import evaluation


# ==========================================
# TEST DATA
# ==========================================

BASELINE = "Classify customer support tickets."
CANDIDATE = "Classify tickets accurately."

CASES = [
    {
        "id": "t01",
        "group": "critical",
        "input": "I was charged twice.",
    },
    {
        "id": "t02",
        "group": "standard",
        "input": "How do I update my profile?",
    },
    {
        "id": "t03",
        "group": "critical",
        "input": "My payment failed.",
    },
]

CONFIG = {
    "model": "test-model",
    "temperature": 0.0,
    "workers": 2,
    "tolerance": 0.05,
    "group_tolerance": 0.10,
    "critical_group": "critical",
}


# ==========================================
# MOCK EVALUATION RESULTS
# ==========================================

def make_results(cases, repeats=1, passed=True):

    return {
        case["id"]: {
            "runs": [
                {
                    "text": '{"priority":"high"}',
                    "passed": passed,
                    "checks": {
                        "valid_json": True,
                        "priority": passed,
                    },
                    "latency": 0.5,
                    "cost": 0.0,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "error": None,
                }
                for _ in range(repeats)
            ],
            "pass_rate": float(passed),
        }
        for case in cases
    }


# ==========================================
# TEST FIXTURE
# ==========================================

@pytest.fixture
def env(tmp_path, monkeypatch):

    root = tmp_path / "project"
    root.mkdir()

    testset = root / "testset"
    testset.mkdir()

    live_results = root / "results" / "live_runs"

    config_path = root / "promptgate.yaml"

    config_path.write_text(
        yaml.safe_dump(CONFIG),
        encoding="utf-8",
    )

    cases_path = testset / "cases.jsonl"

    cases_path.write_text(
        "\n".join(json.dumps(case) for case in CASES),
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluation, "ROOT", root)

    monkeypatch.setattr(
        evaluation,
        "LIVE_RESULTS",
        live_results,
    )

    monkeypatch.setenv("LLM_API_KEY", "fake-test-key")

    monkeypatch.setenv(
        "LLM_BASE_URL",
        "https://example.invalid/v1",
    )

    def fake_suite(
        prompt,
        cases,
        model,
        repeats=1,
        temperature=0.0,
        workers=2,
    ):
        return make_results(cases, repeats)

    suite = Mock(side_effect=fake_suite)

    cache_key = Mock(return_value="test-cache-key")
    cache_load = Mock(return_value=None)
    cache_save = Mock()

    comparison = Mock(
        return_value={
            "verdict": "PASS",
            "reasons": [],
        }
    )

    def fake_html(path, *args):
        Path(path).write_text(
            "<html>Test report</html>",
            encoding="utf-8",
        )

    def fake_markdown(path, *args):
        Path(path).write_text(
            "# Test report",
            encoding="utf-8",
        )

    html = Mock(side_effect=fake_html)
    markdown = Mock(side_effect=fake_markdown)

    monkeypatch.setattr(
        evaluation, "run_suite", suite
    )

    monkeypatch.setattr(
        evaluation, "generate_cache_key", cache_key
    )

    monkeypatch.setattr(
        evaluation, "load_cached_baseline", cache_load
    )

    monkeypatch.setattr(
        evaluation, "save_cached_baseline", cache_save
    )

    monkeypatch.setattr(
        evaluation, "compare", comparison
    )

    monkeypatch.setattr(
        evaluation, "write_html", html
    )

    monkeypatch.setattr(
        evaluation, "write_markdown", markdown
    )

    return {
        "root": root,
        "live_results": live_results,
        "config_path": config_path,
        "cases_path": cases_path,
        "suite": suite,
        "cache_key": cache_key,
        "cache_load": cache_load,
        "cache_save": cache_save,
        "comparison": comparison,
        "html": html,
        "markdown": markdown,
    }


def evaluate(**kwargs):

    return evaluation.evaluate_prompts(
        BASELINE,
        CANDIDATE,
        **kwargs,
    )


# ==========================================
# 1. INPUT VALIDATION
# ==========================================

@pytest.mark.parametrize("prompt", ["", " ", "\n\t"])
def test_empty_baseline(env, prompt):

    with pytest.raises(
        ValueError,
        match="Baseline prompt cannot be empty",
    ):
        evaluation.evaluate_prompts(
            prompt,
            CANDIDATE,
        )

    env["suite"].assert_not_called()


@pytest.mark.parametrize("prompt", ["", " ", "\n\t"])
def test_empty_candidate(env, prompt):

    with pytest.raises(
        ValueError,
        match="Candidate prompt cannot be empty",
    ):
        evaluation.evaluate_prompts(
            BASELINE,
            prompt,
        )

    env["suite"].assert_not_called()


@pytest.mark.parametrize("repeats", [0, -1, -5])
def test_invalid_repeats(env, repeats):

    with pytest.raises(
        ValueError,
        match="Repeats must be at least 1",
    ):
        evaluate(repeats=repeats)

    env["suite"].assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, -10])
def test_invalid_limit(env, limit):

    with pytest.raises(
        ValueError,
        match="Case limit must be at least 1",
    ):
        evaluate(limit=limit)

    env["suite"].assert_not_called()


# ==========================================
# 2. ENVIRONMENT VALIDATION
# ==========================================

def test_missing_api_key(env, monkeypatch):

    monkeypatch.delenv(
        "LLM_API_KEY",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="LLM_API_KEY is not configured",
    ):
        evaluate()

    env["suite"].assert_not_called()


def test_missing_base_url(env, monkeypatch):

    monkeypatch.delenv(
        "LLM_BASE_URL",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="LLM_BASE_URL is not configured",
    ):
        evaluate()

    env["suite"].assert_not_called()


# ==========================================
# 3. CASE SELECTION
# ==========================================

def test_default_case_selection(env):

    result = evaluate()

    assert result["cases"] == CASES
    assert result["configuration"]["cases"] == 3


def test_case_limit_one(env):

    result = evaluate(limit=1)

    assert len(result["cases"]) == 1
    assert result["cases"][0]["id"] == "t01"


def test_case_limit_two(env):

    result = evaluate(limit=2)

    assert len(result["cases"]) == 2


def test_limit_none(env):

    result = evaluate(limit=None)

    assert result["cases"] == CASES


def test_limit_larger_than_dataset(env):

    result = evaluate(limit=100)

    assert len(result["cases"]) == 3


def test_critical_only(env):

    result = evaluate(
        limit=None,
        critical_only=True,
    )

    assert [
        case["id"] for case in result["cases"]
    ] == ["t01", "t03"]

    assert result["configuration"]["critical_only"]


def test_critical_filter_before_limit(env):

    result = evaluate(
        limit=2,
        critical_only=True,
    )

    assert [
        case["id"] for case in result["cases"]
    ] == ["t01", "t03"]


def test_empty_dataset(env):

    env["cases_path"].write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="No evaluation cases found",
    ):
        evaluate()

    env["suite"].assert_not_called()


def test_no_critical_cases(env):

    env["cases_path"].write_text(
        json.dumps(CASES[1]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="No evaluation cases found",
    ):
        evaluate(critical_only=True)


def test_duplicate_case_ids(env):

    env["cases_path"].write_text(
        "\n".join([
            json.dumps(CASES[0]),
            json.dumps(CASES[0]),
        ]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Evaluation case IDs must be unique",
    ):
        evaluate()

    env["suite"].assert_not_called()


def test_custom_critical_group(env):

    config = {
        **CONFIG,
        "critical_group": "standard",
    }

    env["config_path"].write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    result = evaluate(critical_only=True)

    assert [
        case["id"] for case in result["cases"]
    ] == ["t02"]


# ==========================================
# 4. EVALUATION ORCHESTRATION
# ==========================================

def test_both_prompts_evaluated(env):

    evaluate()

    assert env["suite"].call_count == 2


def test_baseline_runs_first(env):

    evaluate()

    calls = env["suite"].call_args_list

    assert calls[0].args[0] == BASELINE
    assert calls[1].args[0] == CANDIDATE


def test_model_forwarded(env):

    evaluate()

    for call in env["suite"].call_args_list:
        assert call.args[2] == "test-model"


def test_repeats_forwarded(env):

    evaluate(repeats=4)

    for call in env["suite"].call_args_list:
        assert call.kwargs["repeats"] == 4


def test_temperature_forwarded(env):

    config = {
        **CONFIG,
        "temperature": 0.4,
    }

    env["config_path"].write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    evaluate()

    for call in env["suite"].call_args_list:
        assert call.kwargs["temperature"] == 0.4


def test_workers_forwarded(env):

    config = {
        **CONFIG,
        "workers": 5,
    }

    env["config_path"].write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    evaluate()

    for call in env["suite"].call_args_list:
        assert call.kwargs["workers"] == 5


def test_same_cases_for_both_prompts(env):

    evaluate(limit=2)

    calls = env["suite"].call_args_list

    assert calls[0].args[1] == calls[1].args[1]
    assert calls[0].args[1] == CASES[:2]


def test_comparison_receives_results(env):

    result = evaluate()

    env["comparison"].assert_called_once_with(
        result["baseline_runs"],
        result["candidate_runs"],
        result["cases"],
        CONFIG,
    )


# ==========================================
# 5. BASELINE CACHE INTEGRATION
# ==========================================

def test_cache_miss(env):

    result = evaluate()

    assert not result["cache"]["baseline_cache_hit"]
    assert env["suite"].call_count == 2


def test_cache_miss_saves_baseline(env):

    result = evaluate()

    env["cache_save"].assert_called_once_with(
        "test-cache-key",
        result["baseline_runs"],
        CASES,
        1,
    )


def test_cache_hit(env):

    cached = make_results(CASES)

    env["cache_load"].return_value = cached

    result = evaluate()

    assert result["cache"]["baseline_cache_hit"]
    assert result["baseline_runs"] == cached


def test_cache_hit_skips_baseline(env):

    env["cache_load"].return_value = make_results(
        CASES
    )

    evaluate()

    env["suite"].assert_called_once()

    assert (
        env["suite"].call_args.args[0]
        == CANDIDATE
    )


def test_cache_hit_does_not_save_again(env):

    env["cache_load"].return_value = make_results(
        CASES
    )

    evaluate()

    env["cache_save"].assert_not_called()


def test_cache_disabled(env):

    result = evaluate(use_cache=False)

    assert not result["cache"]["enabled"]
    assert not result["cache"]["baseline_cache_hit"]
    assert result["cache"]["cache_key"] is None

    assert env["suite"].call_count == 2

    env["cache_key"].assert_not_called()
    env["cache_load"].assert_not_called()
    env["cache_save"].assert_not_called()


def test_cache_key_arguments(env):

    evaluate(
        limit=2,
        repeats=3,
    )

    env["cache_key"].assert_called_once_with(
        BASELINE,
        CASES[:2],
        "test-model",
        3,
        0.0,
        2,
    )


def test_cache_loader_arguments(env):

    evaluate(limit=2, repeats=3)

    env["cache_load"].assert_called_once_with(
        "test-cache-key",
        CASES[:2],
        3,
    )


def test_cache_write_failure_is_nonfatal(env):

    env["cache_save"].side_effect = OSError(
        "Disk write failed"
    )

    result = evaluate()

    assert (
        result["cache"]["warning"]
        == "Disk write failed"
    )

    assert result["comparison"]["verdict"] == "PASS"
    assert env["suite"].call_count == 2


def test_cache_warning_none_on_success(env):

    result = evaluate()

    assert result["cache"]["warning"] is None


# ==========================================
# 6. API EFFICIENCY METRICS
# ==========================================

def test_planned_calls_cache_miss(env):

    result = evaluate(limit=3, repeats=2)

    assert result["cache"]["planned_api_calls"] == 12
    assert result["cache"]["api_calls_saved"] == 0


def test_planned_calls_cache_hit(env):

    env["cache_load"].return_value = make_results(
        CASES,
        repeats=2,
    )

    result = evaluate(limit=3, repeats=2)

    assert result["cache"]["planned_api_calls"] == 6
    assert result["cache"]["api_calls_saved"] == 6


def test_planned_calls_cache_disabled(env):

    result = evaluate(
        limit=2,
        repeats=3,
        use_cache=False,
    )

    assert result["cache"]["planned_api_calls"] == 12
    assert result["cache"]["api_calls_saved"] == 0


def test_planned_calls_critical_only(env):

    result = evaluate(
        critical_only=True,
        repeats=2,
    )

    assert result["cache"]["planned_api_calls"] == 8


# ==========================================
# 7. VERDICT PROPAGATION
# ==========================================

@pytest.mark.parametrize(
    "verdict",
    ["PASS", "FAIL", "ERROR"],
)
def test_verdict_propagation(env, verdict):

    env["comparison"].return_value = {
        "verdict": verdict,
        "reasons": [],
    }

    result = evaluate()

    assert result["comparison"]["verdict"] == verdict


def test_regression_reasons_preserved(env):

    reasons = [
        "Overall score dropped.",
        "Critical case regression.",
    ]

    env["comparison"].return_value = {
        "verdict": "FAIL",
        "reasons": reasons,
    }

    result = evaluate()

    assert result["comparison"]["reasons"] == reasons


# ==========================================
# 8. EXPERIMENT METADATA
# ==========================================

def test_run_id_format(env):

    result = evaluate()

    assert re.fullmatch(
        r"\d{8}_\d{6}_[a-f0-9]{8}",
        result["run_id"],
    )


def test_run_ids_unique(env):

    first = evaluate()
    second = evaluate()

    assert first["run_id"] != second["run_id"]


def test_timestamp_timezone(env):

    result = evaluate()

    timestamp = datetime.fromisoformat(
        result["timestamp"]
    )

    assert timestamp.isoformat().endswith("+00:00")


def test_experiment_source(env):

    result = evaluate()

    assert result["source"] == "live_prompt_lab"


def test_prompts_preserved(env):

    result = evaluate()

    assert result["baseline_prompt"] == BASELINE
    assert result["candidate_prompt"] == CANDIDATE


def test_configuration_preserved(env):

    result = evaluate(repeats=3)

    config = result["configuration"]

    assert config["model"] == "test-model"
    assert config["temperature"] == 0.0
    assert config["workers"] == 2
    assert config["repeats"] == 3
    assert config["cases"] == 3
    assert config["tolerance"] == 0.05
    assert config["group_tolerance"] == 0.10
    assert config["critical_group"] == "critical"


def test_default_configuration_values(env):

    config = {
        "model": "test-model",
        "tolerance": 0.05,
        "group_tolerance": 0.10,
    }

    env["config_path"].write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    result = evaluate()

    config = result["configuration"]

    assert config["temperature"] == 0.0
    assert config["workers"] == 2
    assert config["critical_group"] == "critical"


# ==========================================
# 9. EXPERIMENT PERSISTENCE
# ==========================================

def test_experiment_directory_created(env):

    result = evaluate()

    directory = (
        env["live_results"] / result["run_id"]
    )

    assert directory.is_dir()


@pytest.mark.parametrize(
    "filename",
    [
        "comparison.json",
        "report.html",
        "report.md",
    ],
)
def test_report_file_created(env, filename):

    result = evaluate()

    path = (
        env["live_results"]
        / result["run_id"]
        / filename
    )

    assert path.is_file()
    assert path.stat().st_size > 0


def test_saved_json_matches_payload(env):

    result = evaluate()

    path = (
        env["live_results"]
        / result["run_id"]
        / "comparison.json"
    )

    saved = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert saved == result


def test_previous_experiment_preserved(env):

    first = evaluate()

    first_path = (
        env["live_results"]
        / first["run_id"]
        / "comparison.json"
    )

    original = first_path.read_text(
        encoding="utf-8"
    )

    second = evaluate()

    assert first["run_id"] != second["run_id"]

    assert first_path.read_text(
        encoding="utf-8"
    ) == original


def test_separate_experiment_directories(env):

    evaluate()
    evaluate()

    directories = [
        path
        for path in env["live_results"].iterdir()
        if path.is_dir()
    ]

    assert len(directories) == 2


def test_unicode_prompt_persistence(env):

    baseline = "Classify tickets: भुगतान समस्या"

    result = evaluation.evaluate_prompts(
        baseline,
        CANDIDATE,
    )

    path = (
        env["live_results"]
        / result["run_id"]
        / "comparison.json"
    )

    saved = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert saved["baseline_prompt"] == baseline


# ==========================================
# 10. REPORT GENERATION INTEGRATION
# ==========================================

def test_html_writer_called(env):

    result = evaluate()

    path = (
        env["live_results"]
        / result["run_id"]
        / "report.html"
    )

    env["html"].assert_called_once_with(
        str(path),
        result["comparison"],
        result["cases"],
        result["baseline_runs"],
        result["candidate_runs"],
    )


def test_markdown_writer_called(env):

    result = evaluate()

    path = (
        env["live_results"]
        / result["run_id"]
        / "report.md"
    )

    env["markdown"].assert_called_once_with(
        str(path),
        result["comparison"],
        result["cases"],
        result["baseline_runs"],
        result["candidate_runs"],
    )


# ==========================================
# 11. PROGRESS CALLBACKS
# ==========================================

def test_progress_callback_invoked(env):

    callback = Mock()

    evaluate(progress_callback=callback)

    assert callback.call_count > 0


def test_progress_starts_correctly(env):

    callback = Mock()

    evaluate(progress_callback=callback)

    assert callback.call_args_list[0].args == (
        0.05,
        "Loading evaluation configuration...",
    )


def test_progress_finishes_at_one(env):

    callback = Mock()

    evaluate(progress_callback=callback)

    assert callback.call_args_list[-1].args == (
        1.0,
        "Evaluation complete.",
    )


def test_progress_is_monotonic(env):

    callback = Mock()

    evaluate(progress_callback=callback)

    values = [
        call.args[0]
        for call in callback.call_args_list
    ]

    assert values == sorted(values)


def test_progress_values_in_range(env):

    callback = Mock()

    evaluate(progress_callback=callback)

    for call in callback.call_args_list:
        assert 0.0 <= call.args[0] <= 1.0


def test_cache_hit_progress_message(env):

    env["cache_load"].return_value = make_results(
        CASES
    )

    callback = Mock()

    evaluate(progress_callback=callback)

    messages = [
        call.args[1]
        for call in callback.call_args_list
    ]

    assert any(
        "cache hit" in message.lower()
        for message in messages
    )


def test_cache_miss_progress_message(env):

    callback = Mock()

    evaluate(progress_callback=callback)

    messages = [
        call.args[1]
        for call in callback.call_args_list
    ]

    assert "Running baseline prompt..." in messages


def test_no_progress_callback(env):

    result = evaluate(progress_callback=None)

    assert result["comparison"]["verdict"] == "PASS"


# ==========================================
# 12. ERROR HANDLING
# ==========================================

def test_missing_configuration(env):

    env["config_path"].unlink()

    with pytest.raises(FileNotFoundError):
        evaluate()

    env["suite"].assert_not_called()


def test_missing_dataset(env):

    env["cases_path"].unlink()

    with pytest.raises(FileNotFoundError):
        evaluate()

    env["suite"].assert_not_called()


def test_invalid_json_dataset(env):

    env["cases_path"].write_text(
        "invalid json",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        evaluate()

    env["suite"].assert_not_called()


def test_suite_exception_propagates(env):

    env["suite"].side_effect = RuntimeError(
        "Evaluation failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Evaluation failed",
    ):
        evaluate()

    assert not env["live_results"].exists()


def test_comparison_exception_propagates(env):

    env["comparison"].side_effect = RuntimeError(
        "Comparison failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Comparison failed",
    ):
        evaluate()

    assert not env["live_results"].exists()


def test_invalid_input_creates_no_experiment(env):

    with pytest.raises(ValueError):
        evaluation.evaluate_prompts(
            "",
            CANDIDATE,
        )

    assert not env["live_results"].exists()
