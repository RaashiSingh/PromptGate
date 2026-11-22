
import html


def _short(s, n):
    """Shorten long text for report tables."""
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def _metric_rows(res):
    """Build comparison rows for report metrics."""
    bs, cs = res["base"], res["cand"]

    rows = [
        (
            "Overall score",
            f'{bs["score"]:.1%}',
            f'{cs["score"]:.1%}',
            f'{(cs["score"] - bs["score"]) * 100:+.1f} pts',
        ),
        (
            "Estimated cost (USD)",
            f'${bs["cost"]:.4f}',
            f'${cs["cost"]:.4f}',
            f'{cs["cost"] - bs["cost"]:+.4f}',
        ),
        (
            "Avg latency (s)",
            f'{bs["latency"]:.2f}',
            f'{cs["latency"]:.2f}',
            f'{cs["latency"] - bs["latency"]:+.2f}',
        ),
    ]

    for group in bs["groups"]:
        rows.append(
            (
                f"Group: {group}",
                f'{bs["groups"][group]:.1%}',
                f'{cs["groups"][group]:.1%}',
                f'{(cs["groups"][group] - bs["groups"][group]) * 100:+.1f} pts',
            )
        )

    return rows


def _flip_rows(ids, cases, base, cand):
    """Build detailed rows for changed test cases."""
    case_lookup = {c["id"]: c for c in cases}
    rows = []

    for case_id in ids:
        case = case_lookup[case_id]

        rows.append(
            (
                case_id,
                case["group"],
                _short(case["input"], 110),
                f'{case["priority"]} / {case["category"]}',
                _short(base[case_id]["runs"][0]["text"], 160),
                _short(cand[case_id]["runs"][0]["text"], 160),
                f'{base[case_id]["pass_rate"]:.0%} → '
                f'{cand[case_id]["pass_rate"]:.0%}',
            )
        )

    return rows


HEAD = [
    "ID",
    "Group",
    "Ticket",
    "Expected",
    "Baseline output",
    "Candidate output",
    "Pass rate",
]


def write_html(path, res, cases, base, cand):
    """Generate a standalone HTML regression report."""

    colors = {
        "PASS": "#1a7f37",
        "FAIL": "#cf222e",
        "ERROR": "#9a6700",
    }

    color = colors[res["verdict"]]

    def table(headers, rows):
        heading = "".join(
            f"<th>{html.escape(str(x))}</th>"
            for x in headers
        )

        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in row
            )
            + "</tr>"
            for row in rows
        )

        return f"<table><tr>{heading}</tr>{body}</table>"

    reasons = "".join(
        f"<li>{html.escape(reason)}</li>"
        for reason in res["reasons"]
    ) or "<li>No regression detected</li>"

    metrics = table(
        ["Metric", "Baseline", "Candidate", "Change"],
        _metric_rows(res),
    )

    down = _flip_rows(res["flips_down"], cases, base, cand)
    up = _flip_rows(res["flips_up"], cases, base, cand)

    down_html = table(HEAD, down) if down else "<p>None</p>"
    up_html = table(HEAD, up) if up else "<p>None</p>"

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PromptGate Regression Report</title>
<style>
body {{
    font-family: system-ui, sans-serif;
    max-width: 1100px;
    margin: 2rem auto;
    padding: 0 1rem;
    line-height: 1.5;
    color: #24292f;
}}
.badge {{
    background: {color};
    color: white;
    padding: .25rem .8rem;
    border-radius: 6px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: .5rem 0 1.5rem;
}}
td, th {{
    border: 1px solid #ccc;
    padding: .5rem;
    font-size: .85rem;
    vertical-align: top;
    text-align: left;
    overflow-wrap: anywhere;
}}
th {{
    background: #f3f3f3;
}}
</style>
</head>
<body>
<h1>PromptGate <span class="badge">{res["verdict"]}</span></h1>
<ul>{reasons}</ul>
<h2>Performance Metrics</h2>
{metrics}
<h2>Cases that flipped to FAIL</h2>
{down_html}
<h2>Cases that flipped to PASS</h2>
{up_html}
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as file:
        file.write(doc)


def write_markdown(path, res, cases, base, cand):
    """Generate a Markdown report for GitHub PR comments."""

    icons = {
        "PASS": "PASS",
        "FAIL": "FAIL",
        "ERROR": "ERROR",
    }

    lines = [
        f'## PromptGate: {icons[res["verdict"]]}',
        "",
    ]

    for reason in res["reasons"] or ["No regression detected"]:
        lines.append(f"- {reason}")

    lines += [
        "",
        "| Metric | Baseline | Candidate | Change |",
        "|---|---|---|---|",
    ]

    for row in _metric_rows(res):
        lines.append(
            "| " + " | ".join(row) + " |"
        )

    down = _flip_rows(res["flips_down"], cases, base, cand)

    if down:
        lines += [
            "",
            "**Flipped to FAIL**",
            "",
            "| ID | Group | Expected | Candidate output | Pass rate |",
            "|---|---|---|---|---|",
        ]

        for row in down:
            output = row[5].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row[0]} | {row[1]} | {row[3]} | "
                f"`{output}` | {row[6]} |"
            )

    if res["flips_up"]:
        lines += [
            "",
            "**Flipped to PASS:** "
            + ", ".join(res["flips_up"]),
        ]

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
