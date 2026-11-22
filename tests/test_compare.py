
import pytest

from promptgate.compare import compare


# ==========================================
# TEST CONFIGURATION
# ==========================================

CFG = {
    "tolerance": 0.05,
    "group_tolerance": 0.10,
    "critical_group": "critical",
}

CASES = [
    {"id": "t01", "group": "critical"},
    {"id": "t02", "group": "critical"},
    {"id": "t03", "group": "standard"},
    {"id": "t04", "group": "standard"},
]


# ==========================================
# MOCK EVALUATION RESULTS
# ==========================================

def make_results(rates, errors=None):
    errors = errors or {}
    results = {}

    for case_id, rate in rates.items():
        total_runs = 10
        passed_runs = round(rate * total_runs)

        runs = []

        for index in range(total_runs):
            runs.append({
                "text": "Mock response",
                "passed": index < passed_runs,
                "checks": {},
                "error": (
                    errors.get(case_id)
                    if index == 0
                    else None
                ),
                "latency": 0.5,
                "cost": 0.0,
                "input_tokens": 100,
                "output_tokens": 50,
            })

        results[case_id] = {
            "runs": runs,
            "pass_rate": passed_runs / total_runs,
        }

    return results


def perfect_results():
    return make_results({
        "t01": 1.0,
        "t02": 1.0,
        "t03": 1.0,
        "t04": 1.0,
    })


# ==========================================
# PASS VERDICT TESTS
# ==========================================

def test_identical_results_pass():
    base = perfect_results()
    cand = perfect_results()

    result = compare(base, cand, CASES, CFG)

    assert result["verdict"] == "PASS"
    assert result["reasons"] == []
    assert result["flips_down"] == []


def test_candidate_improvement_passes():
    base = make_results({
        "t01": 0.4,
        "t02": 1.0,
        "t03": 1.0,
        "t04": 1.0,
    })

    cand = perfect_results()

    result = compare(base, cand, CASES, CFG)

    assert result["verdict"] == "PASS"
    assert "t01" in result["flips_up"]


def test_small_drop_within_tolerance():
    base = perfect_results()

    cand = make_results({
        "t01": 1.0,
        "t02": 1.0,
        "t03": 0.9,
        "t04": 1.0,
    })

    result = compare(base, cand, CASES, CFG)

    assert result["verdict"] == "PASS"


def test_existing_failure_not_new_regression():
    rates = {
        "t01": 0.0,
        "t02": 1.0,
        "t03": 1.0,
        "t04": 1.0,
    }

    result = compare(
        make_results(rates),
        make_results(rates),
        CASES,
        CFG,
    )

    assert result["verdict"] == "PASS"
    assert result["flips_down"] == []


# ==========================================
# FAIL VERDICT TESTS
# ==========================================

def test_overall_score_regression():
    base = perfect_results()

    cand = make_results({
        "t01": 0.8,
        "t02": 0.8,
        "t03": 0.8,
        "t04": 0.8,
    })

    result = compare(base, cand, CASES, CFG)

    assert result["verdict"] == "FAIL"

    assert any(
        "Overall score dropped" in reason
        for reason in result["reasons"]
    )


def test_group_score_regression():
    base = perfect_results()

    cand = make_results({
        "t01": 1.0,
        "t02": 1.0,
        "t03": 0.6,
        "t04": 0.6,
    })

    result = compare(base, cand, CASES, CFG)

    assert result["verdict"] == "FAIL"

    assert any(
        "Group 'standard'" in reason
        for reason in result["reasons"]
    )


def test_critical_regression_overrides_tolerance():
    base = perfect_results()

    cand = make_results({
        "t01": 0.0,
        "t02": 1.0,
        "t03": 1.0,
        "t04": 1.0,
    })

    relaxed_cfg = {
        **CFG,
        "tolerance": 1.0,
        "group_tolerance": 1.0,
    }

    result = compare(
        base, cand, CASES, relaxed_cfg
    )

    assert result["verdict"] == "FAIL"
    assert "t01" in result["flips_down"]

    assert any(
        "critical case(s)" in reason
        for reason in result["reasons"]
    )


def test_standard_flip_with_relaxed_thresholds():
    base = perfect_results()

    cand = make_results({
        "t01": 1.0,
        "t02": 1.0,
        "t03": 0.0,
        "t04": 1.0,
    })

    relaxed_cfg = {
        **CFG,
        "tolerance": 1.0,
        "group_tolerance": 1.0,
    }

    result = compare(
        base, cand, CASES, relaxed_cfg
    )

    assert result["verdict"] == "PASS"
    assert "t03" in result["flips_down"]


def test_half_pass_rate_is_considered_passing():
    base = perfect_results()

    cand = make_results({
        "t01": 0.5,
        "t02": 1.0,
        "t03": 1.0,
        "t04": 1.0,
    })

    relaxed_cfg = {
        **CFG,
        "tolerance": 1.0,
        "group_tolerance": 1.0,
    }

    result = compare(
        base, cand, CASES, relaxed_cfg
    )

    assert result["verdict"] == "PASS"
    assert "t01" not in result["flips_down"]


# ==========================================
# API ERROR TESTS
# ==========================================

@pytest.mark.parametrize("error_side", [
    "baseline",
    "candidate",
    "both",
])
def test_api_errors_return_error(error_side):
    rates = {
        "t01": 0.9,
        "t02": 1.0,
        "t03": 1.0,
        "t04": 1.0,
    }

    base = perfect_results()
    cand = perfect_results()

    if error_side in ("baseline", "both"):
        base = make_results(
            rates,
            {"t01": "API rate limit exceeded"}
        )

    if error_side in ("candidate", "both"):
        cand = make_results(
            rates,
            {"t01": "API request failed"}
        )

    result = compare(base, cand, CASES, CFG)

    assert result["verdict"] == "ERROR"
    assert "API errors:" in result["reasons"][0]


# ==========================================
# METRIC ACCURACY TESTS
# ==========================================

def test_summary_metrics():
    base = perfect_results()

    result = compare(base, base, CASES, CFG)

    summary = result["base"]

    assert summary["score"] == 1.0
    assert summary["calls"] == 40
    assert summary["errors"] == 0
    assert summary["latency"] == 0.5
    assert summary["cost"] == 0.0
    assert summary["input_tokens"] == 4000
    assert summary["output_tokens"] == 2000


def test_group_metrics():
    base = perfect_results()

    cand = make_results({
        "t01": 1.0,
        "t02": 0.8,
        "t03": 0.6,
        "t04": 0.4,
    })

    result = compare(base, cand, CASES, CFG)

    assert result["cand"]["score"] == pytest.approx(0.7)

    assert result["cand"]["groups"]["critical"] == (
        pytest.approx(0.9)
    )

    assert result["cand"]["groups"]["standard"] == (
        pytest.approx(0.5)
    )
