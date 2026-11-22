
import pytest

from promptgate.report import (
    _short,
    _metric_rows,
    _flip_rows,
    write_html,
    write_markdown,
)


# ==========================================
# TEST FIXTURES
# ==========================================

@pytest.fixture
def cases():
    return [
        {
            "id": "t01",
            "group": "critical",
            "input": "I was charged twice.",
            "priority": "high",
            "category": "billing",
        },
        {
            "id": "t02",
            "group": "standard",
            "input": "How do I change my password?",
            "priority": "low",
            "category": "technical",
        },
    ]


def make_result(text, pass_rate):
    return {
        "runs": [{"text": text}],
        "pass_rate": pass_rate,
    }


@pytest.fixture
def baseline():
    return {
        "t01": make_result(
            '{"priority":"high","category":"billing"}',
            1.0,
        ),
        "t02": make_result(
            '{"priority":"low","category":"other"}',
            0.0,
        ),
    }


@pytest.fixture
def candidate():
    return {
        "t01": make_result(
            '{"priority":"low","category":"billing"}',
            0.0,
        ),
        "t02": make_result(
            '{"priority":"low","category":"technical"}',
            1.0,
        ),
    }


@pytest.fixture
def comparison():
    return {
        "verdict": "FAIL",
        "reasons": ["Critical case regression detected"],
        "flips_down": ["t01"],
        "flips_up": ["t02"],
        "base": {
            "score": 0.8,
            "cost": 0.002,
            "latency": 0.8,
            "groups": {
                "critical": 1.0,
                "standard": 0.6,
            },
        },
        "cand": {
            "score": 0.6,
            "cost": 0.003,
            "latency": 1.0,
            "groups": {
                "critical": 0.5,
                "standard": 0.7,
            },
        },
    }


# ==========================================
# TEXT SHORTENING TESTS
# ==========================================

@pytest.mark.parametrize(
    "value,limit,expected",
    [
        ("hello", 10, "hello"),
        ("hello", 5, "hello"),
        ("", 10, ""),
        ("   ", 10, ""),
        ("hello   world", 20, "hello world"),
        ("hello\nworld", 20, "hello world"),
        ("hello\tworld", 20, "hello world"),
        ("  hello  ", 20, "hello"),
        ("a", 1, "a"),
        ("नमस्ते", 20, "नमस्ते"),
    ],
)
def test_short_text(value, limit, expected):
    assert _short(value, limit) == expected


@pytest.mark.parametrize("limit", [1, 3, 5, 10])
def test_short_truncates_long_text(limit):
    result = _short("abcdefghijklmnopqrstuvwxyz", limit)

    assert result == "abcdefghijklmnopqrstuvwxyz"[:limit] + "\u2026"


def test_short_zero_limit():
    assert _short("hello", 0) == "\u2026"


# ==========================================
# METRIC ROW TESTS
# ==========================================

def test_metric_row_count(comparison):
    assert len(_metric_rows(comparison)) == 5


def test_overall_score_row(comparison):
    assert _metric_rows(comparison)[0] == (
        "Overall score",
        "80.0%",
        "60.0%",
        "-20.0 pts",
    )


def test_cost_row(comparison):
    assert _metric_rows(comparison)[1] == (
        "Estimated cost (USD)",
        "$0.0020",
        "$0.0030",
        "+0.0010",
    )


def test_latency_row(comparison):
    assert _metric_rows(comparison)[2] == (
        "Avg latency (s)",
        "0.80",
        "1.00",
        "+0.20",
    )


def test_critical_group_row(comparison):
    assert _metric_rows(comparison)[3] == (
        "Group: critical",
        "100.0%",
        "50.0%",
        "-50.0 pts",
    )


def test_standard_group_row(comparison):
    assert _metric_rows(comparison)[4] == (
        "Group: standard",
        "60.0%",
        "70.0%",
        "+10.0 pts",
    )


def test_no_groups(comparison):
    comparison["base"]["groups"] = {}
    comparison["cand"]["groups"] = {}

    assert len(_metric_rows(comparison)) == 3


def test_equal_scores(comparison):
    comparison["cand"]["score"] = 0.8

    assert _metric_rows(comparison)[0][3] == "+0.0 pts"


def test_improved_score(comparison):
    comparison["cand"]["score"] = 0.9

    assert _metric_rows(comparison)[0][3] == "+10.0 pts"


def test_zero_cost(comparison):
    comparison["base"]["cost"] = 0.0
    comparison["cand"]["cost"] = 0.0

    assert _metric_rows(comparison)[1] == (
        "Estimated cost (USD)",
        "$0.0000",
        "$0.0000",
        "+0.0000",
    )


# ==========================================
# CASE FLIP TESTS
# ==========================================

def test_empty_flip_rows(cases, baseline, candidate):
    assert _flip_rows(
        [], cases, baseline, candidate
    ) == []


def test_regression_row(cases, baseline, candidate):
    row = _flip_rows(
        ["t01"], cases, baseline, candidate
    )[0]

    assert row[0] == "t01"
    assert row[1] == "critical"
    assert row[2] == "I was charged twice."
    assert row[3] == "high / billing"
    assert row[4] == baseline["t01"]["runs"][0]["text"]
    assert row[5] == candidate["t01"]["runs"][0]["text"]
    assert row[6] == "100% \u2192 0%"


def test_improvement_row(cases, baseline, candidate):
    row = _flip_rows(
        ["t02"], cases, baseline, candidate
    )[0]

    assert row[6] == "0% \u2192 100%"


def test_flip_row_order(cases, baseline, candidate):
    rows = _flip_rows(
        ["t02", "t01"],
        cases,
        baseline,
        candidate,
    )

    assert [row[0] for row in rows] == ["t02", "t01"]


def test_long_ticket_truncation(cases, baseline, candidate):
    cases[0]["input"] = "x" * 300

    row = _flip_rows(
        ["t01"], cases, baseline, candidate
    )[0]

    assert row[2] == "x" * 110 + "\u2026"


def test_long_baseline_output(cases, baseline, candidate):
    baseline["t01"]["runs"][0]["text"] = "x" * 300

    row = _flip_rows(
        ["t01"], cases, baseline, candidate
    )[0]

    assert row[4] == "x" * 160 + "\u2026"


def test_long_candidate_output(cases, baseline, candidate):
    candidate["t01"]["runs"][0]["text"] = "y" * 300

    row = _flip_rows(
        ["t01"], cases, baseline, candidate
    )[0]

    assert row[5] == "y" * 160 + "\u2026"


def test_unknown_case_id(cases, baseline, candidate):
    with pytest.raises(KeyError):
        _flip_rows(
            ["unknown"],
            cases,
            baseline,
            candidate,
        )


# ==========================================
# REPORT GENERATION HELPERS
# ==========================================

def generate_html(
    tmp_path, comparison, cases, baseline, candidate
):
    path = tmp_path / "report.html"

    write_html(
        path, comparison, cases, baseline, candidate
    )

    return path.read_text(encoding="utf-8")


def generate_markdown(
    tmp_path, comparison, cases, baseline, candidate
):
    path = tmp_path / "report.md"

    write_markdown(
        path, comparison, cases, baseline, candidate
    )

    return path.read_text(encoding="utf-8")


# ==========================================
# HTML REPORT TESTS
# ==========================================

@pytest.mark.parametrize(
    "verdict,color",
    [
        ("PASS", "#1a7f37"),
        ("FAIL", "#cf222e"),
        ("ERROR", "#9a6700"),
    ],
)
def test_html_verdict(
    tmp_path, comparison, cases, baseline,
    candidate, verdict, color
):
    comparison["verdict"] = verdict

    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert f'<span class="badge">{verdict}</span>' in content
    assert color in content


def test_html_document_structure(
    tmp_path, comparison, cases, baseline, candidate
):
    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert content.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in content
    assert "<title>PromptGate Regression Report</title>" in content
    assert content.endswith("</html>")


@pytest.mark.parametrize(
    "expected",
    [
        "Performance Metrics",
        "Cases that flipped to FAIL",
        "Cases that flipped to PASS",
        "80.0%",
        "60.0%",
        "-20.0 pts",
        "$0.0020",
        "$0.0030",
        "Group: critical",
        "Group: standard",
        "Critical case regression detected",
        "t01",
        "t02",
        "100% \u2192 0%",
        "0% \u2192 100%",
    ],
)
def test_html_contains_expected_content(
    tmp_path, comparison, cases, baseline,
    candidate, expected
):
    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert expected in content


def test_html_no_regressions(
    tmp_path, comparison, cases, baseline, candidate
):
    comparison["reasons"] = []
    comparison["flips_down"] = []
    comparison["flips_up"] = []
    comparison["verdict"] = "PASS"

    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "No regression detected" in content
    assert content.count("<p>None</p>") == 2


# ==========================================
# HTML ESCAPING AND SECURITY TESTS
# ==========================================

@pytest.mark.parametrize(
    "raw,escaped",
    [
        ("<script>test</script>",
         "&lt;script&gt;test&lt;/script&gt;"),
        ("<b>hello</b>",
         "&lt;b&gt;hello&lt;/b&gt;"),
        ("a & b", "a &amp; b"),
        ('"hello"', "&quot;hello&quot;"),
        ("it's", "it&#x27;s"),
    ],
)
def test_html_escapes_candidate_output(
    tmp_path, comparison, cases, baseline,
    candidate, raw, escaped
):
    candidate["t01"]["runs"][0]["text"] = raw

    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert escaped in content
    assert raw not in content


def test_html_escapes_baseline_output(
    tmp_path, comparison, cases, baseline, candidate
):
    baseline["t01"]["runs"][0]["text"] = "<test>baseline</test>"

    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "&lt;test&gt;baseline&lt;/test&gt;" in content
    assert "<test>baseline</test>" not in content


def test_html_escapes_ticket(
    tmp_path, comparison, cases, baseline, candidate
):
    cases[0]["input"] = "<test>ticket</test>"

    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "&lt;test&gt;ticket&lt;/test&gt;" in content


def test_html_escapes_reasons(
    tmp_path, comparison, cases, baseline, candidate
):
    comparison["reasons"] = ["<test>reason</test>"]

    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "&lt;test&gt;reason&lt;/test&gt;" in content


def test_html_unicode_output(
    tmp_path, comparison, cases, baseline, candidate
):
    candidate["t01"]["runs"][0]["text"] = "नमस्ते 日本語 café"

    content = generate_html(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "नमस्ते 日本語 café" in content


# ==========================================
# MARKDOWN REPORT TESTS
# ==========================================

@pytest.mark.parametrize("verdict", ["PASS", "FAIL", "ERROR"])
def test_markdown_verdict(
    tmp_path, comparison, cases, baseline,
    candidate, verdict
):
    comparison["verdict"] = verdict

    content = generate_markdown(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert content.startswith(f"## PromptGate: {verdict}")


@pytest.mark.parametrize(
    "expected",
    [
        "| Metric | Baseline | Candidate | Change |",
        "|---|---|---|---|",
        "80.0%",
        "60.0%",
        "-20.0 pts",
        "$0.0020",
        "$0.0030",
        "Group: critical",
        "Group: standard",
        "Critical case regression detected",
        "**Flipped to FAIL**",
        "**Flipped to PASS:** t02",
        "high / billing",
        "100% \u2192 0%",
    ],
)
def test_markdown_contains_expected_content(
    tmp_path, comparison, cases, baseline,
    candidate, expected
):
    content = generate_markdown(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert expected in content


def test_markdown_no_regressions(
    tmp_path, comparison, cases, baseline, candidate
):
    comparison["verdict"] = "PASS"
    comparison["reasons"] = []
    comparison["flips_down"] = []
    comparison["flips_up"] = []

    content = generate_markdown(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "No regression detected" in content
    assert "**Flipped to FAIL**" not in content
    assert "**Flipped to PASS:**" not in content


def test_markdown_pipe_escaping(
    tmp_path, comparison, cases, baseline, candidate
):
    candidate["t01"]["runs"][0]["text"] = "left|right"

    content = generate_markdown(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "left\\|right" in content


def test_markdown_multiline_output(
    tmp_path, comparison, cases, baseline, candidate
):
    candidate["t01"]["runs"][0]["text"] = (
        "first line\nsecond line"
    )

    content = generate_markdown(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "first line second line" in content


def test_markdown_unicode(
    tmp_path, comparison, cases, baseline, candidate
):
    candidate["t01"]["runs"][0]["text"] = "नमस्ते 日本語 café"

    content = generate_markdown(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert "नमस्ते 日本語 café" in content


def test_markdown_trailing_newline(
    tmp_path, comparison, cases, baseline, candidate
):
    content = generate_markdown(
        tmp_path, comparison, cases, baseline, candidate
    )

    assert content.endswith("\n")


# ==========================================
# FILE HANDLING TESTS
# ==========================================

@pytest.mark.parametrize(
    "writer,filename",
    [
        (write_html, "report.html"),
        (write_markdown, "report.md"),
    ],
)
def test_report_file_creation(
    tmp_path, comparison, cases, baseline,
    candidate, writer, filename
):
    path = tmp_path / filename

    assert not path.exists()

    writer(path, comparison, cases, baseline, candidate)

    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.parametrize(
    "writer,filename",
    [
        (write_html, "report.html"),
        (write_markdown, "report.md"),
    ],
)
def test_report_overwrites_existing_file(
    tmp_path, comparison, cases, baseline,
    candidate, writer, filename
):
    path = tmp_path / filename
    path.write_text("OLD CONTENT", encoding="utf-8")

    writer(path, comparison, cases, baseline, candidate)

    content = path.read_text(encoding="utf-8")

    assert "OLD CONTENT" not in content
    assert "PromptGate" in content


@pytest.mark.parametrize("writer", [write_html, write_markdown])
def test_invalid_verdict(
    tmp_path, comparison, cases, baseline,
    candidate, writer
):
    comparison["verdict"] = "UNKNOWN"

    with pytest.raises(KeyError):
        writer(
            tmp_path / "report",
            comparison,
            cases,
            baseline,
            candidate,
        )


@pytest.mark.parametrize("writer", [write_html, write_markdown])
def test_missing_parent_directory(
    tmp_path, comparison, cases, baseline,
    candidate, writer
):
    path = tmp_path / "missing" / "report"

    with pytest.raises(FileNotFoundError):
        writer(path, comparison, cases, baseline, candidate)
