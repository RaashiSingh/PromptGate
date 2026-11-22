import sys
import json
import difflib
from pathlib import Path

import pandas as pd
import streamlit as st


# ==========================================
# PROJECT SETUP
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from dashboard.data_loader import (
    get_available_benchmarks,
    load_benchmark,
    calculate_metrics,
    build_heatmap_data,
    load_mutation_report,
    load_test_cases,
    load_prompt_pair,
)

from dashboard.prompt_lab import render_prompt_lab, render_live_registry

from dashboard.components import (
    render_kpis,
    outcome_chart,
    mutation_impact_chart,
    efficiency_chart,
    regression_heatmap,
    duration_chart,
    format_percent,
)


st.set_page_config(
    page_title="PromptGate | Evaluation Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# LOAD CUSTOM STYLES
# ==========================================

css_path = Path(__file__).parent / "styles.css"

if css_path.exists():

    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )


# ==========================================
# SIDEBAR
# ==========================================

benchmarks = get_available_benchmarks()

if not benchmarks:

    st.error("No saved benchmark reports found.")
    st.stop()


with st.sidebar:

    st.title("🛡️ PromptGate")

    st.caption("LLM Evaluation Studio")

    st.divider()

    st.caption("EXPERIMENT CONFIGURATION")

    versions = list(benchmarks.keys())

    version = st.selectbox(
        "Benchmark version",
        versions,
        index=len(versions) - 1
    )

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Prompt Lab",
            "Mutation Explorer",
            "Regression Heatmap",
            "Efficiency Analytics",
            "Experiment Registry",
        ]
    )

    st.divider()

    st.caption("BENCHMARK MODEL")

    st.code(
        "openai/gpt-oss-20b",
        language=None
    )

    st.caption("Inference provider: Groq")

    st.divider()

    st.caption("PromptGate • Development build")


# ==========================================
# LOAD BENCHMARK
# ==========================================

try:

    df = load_benchmark(version)

    if df.empty:
        st.warning("This benchmark has no mutations.")
        st.stop()

    metrics = calculate_metrics(df)

except Exception as error:

    st.error(
        f"Unable to load benchmark: {error}"
    )

    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">
        AI ENGINEERING / EVALUATION INTELLIGENCE
    </div>
    <h1>PromptGate</h1>
    <p>
        Detect regressions. Validate changes.
        Ship prompts with confidence.
    </p>
</div>
""", unsafe_allow_html=True)

case_counts = (
    df["cases"].dropna().unique().tolist()
    if "cases" in df.columns
    else []
)

case_scope = ", ".join(
    str(int(count))
    for count in sorted(case_counts)
)

st.caption(
    f"BENCHMARK {version}  |  "
    f"{len(df)} mutations  |  "
    f"Cases per mutation: {case_scope or 'N/A'}  |  "
    "Saved experimental results"
)


# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":

    st.subheader("Evaluation overview")

    render_kpis(metrics)

    st.divider()

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            outcome_chart(df),
            width="stretch"
        )

    with right:

        st.plotly_chart(
            efficiency_chart(metrics),
            width="stretch"
        )

    st.subheader("Mutation fingerprint")

    st.caption(
        "FAIL means the gate rejected a prompt. "
        "An expected rejection is a successful "
        "regression detection."
    )

    columns = st.columns(7)

    for index, row in df.iterrows():

        matched = (
            row["verdict"]
            == row["expected_verdict"]
        )

        status = "✓" if matched else "⚠"

        with columns[index % 7]:

            if st.button(
                f"{status} {row['mutation_id']}",
                key=f"fingerprint_{version}_{index}",
                help=row["name"],
                width="stretch"
            ):

                st.session_state[
                    "selected_mutation"
                ] = row["mutation_id"]

            st.caption(row["verdict"])

    mutation_ids = df["mutation_id"].tolist()

    selected = st.session_state.get(
        "selected_mutation",
        mutation_ids[0]
    )

    if selected not in mutation_ids:
        selected = mutation_ids[0]

    row = df.loc[
        df["mutation_id"] == selected
    ].iloc[0]

    with st.container(border=True):

        st.markdown(
            f"**{selected} — {row['name']}**"
        )

        st.caption(row["outcome"])

        a, b, c = st.columns(3)

        a.metric(
            "Baseline",
            format_percent(
                row.get("baseline_score")
            )
        )

        b.metric(
            "Candidate",
            format_percent(
                row.get("candidate_score")
            )
        )

        c.metric(
            "Score drop",
            format_percent(
                row.get("score_drop")
            )
        )

    if "candidate_score" in df.columns:

        st.plotly_chart(
            mutation_impact_chart(df),
            width="stretch"
        )

    st.caption(
        "Results apply to the selected benchmark "
        "and do not establish general model reliability."
    )


# ==========================================
# PROMPT LAB
# ==========================================

elif page == "Prompt Lab":
    render_prompt_lab()


# ==========================================
# MUTATION EXPLORER
# ==========================================

elif page == "Mutation Explorer":

    st.subheader("Mutation Explorer")

    available_types = [
        "All",
        *sorted(df["type"].unique())
    ]

    selected_type = st.selectbox(
        "Filter by mutation type",
        available_types
    )

    filtered = df.copy()

    if selected_type != "All":

        filtered = filtered[
            filtered["type"] == selected_type
        ]

    st.dataframe(
        filtered[
            [
                "mutation_id",
                "name",
                "type",
                "verdict",
                "outcome"
            ]
        ],
        hide_index=True,
        width="stretch"
    )

    mutation_id = st.selectbox(
        "Inspect mutation",
        filtered["mutation_id"].tolist()
    )

    report = load_mutation_report(
        version,
        mutation_id
    )

    if report is None:

        st.warning("Report not available.")
        st.stop()

    comparison = report["comparison"]

    a, b, c = st.columns(3)

    a.metric(
        "Verdict",
        comparison["verdict"]
    )

    b.metric(
        "Baseline score",
        format_percent(
            comparison["base"]["score"]
        )
    )

    c.metric(
        "Candidate score",
        format_percent(
            comparison["cand"]["score"]
        )
    )

    st.subheader("Gate decision")

    if comparison["reasons"]:

        for reason in comparison["reasons"]:

            st.write(f"• {reason}")

    else:

        st.success(
            "No regression threshold was violated."
        )

    st.subheader("Response inspector")

    baseline = report.get(
        "baseline_runs", {}
    )

    candidate = report.get(
        "candidate_runs", {}
    )

    shared_cases = sorted(
        set(baseline) & set(candidate)
    )

    if not shared_cases:

        st.info("No comparable responses found.")
        st.stop()

    case_id = st.selectbox(
        "Test case",
        shared_cases
    )

    cases = load_test_cases()

    if case_id in cases:

        st.markdown("**Test input**")

        st.write(
            cases[case_id].get("input", "")
        )

    left, right = st.columns(2)

    for column, label, result in [
        (
            left,
            "Baseline response",
            baseline[case_id]
        ),
        (
            right,
            "Candidate response",
            candidate[case_id]
        ),
    ]:

        with column:

            st.markdown(f"**{label}**")

            st.metric(
                "Pass rate",
                format_percent(
                    result["pass_rate"]
                )
            )

            for index, run in enumerate(
                result["runs"],
                start=1
            ):

                with st.expander(
                    f"Run {index}",
                    expanded=True
                ):

                    if run.get("error"):

                        st.error(run["error"])

                    elif run.get("passed"):

                        st.success("Assertions passed")

                    else:

                        st.error("Assertions failed")

                    response_text = run.get(
                        "text", ""
                    )

                    try:

                        st.json(
                            json.loads(response_text)
                        )

                    except (
                        json.JSONDecodeError,
                        TypeError
                    ):

                        st.code(
                            response_text,
                            language=None
                        )

                    st.markdown("**Assertions**")

                    for check, passed in run.get(
                        "checks", {}
                    ).items():

                        status = (
                            "PASS" if passed else "FAIL"
                        )

                        st.write(
                            f"{check}: {status}"
                        )

                    st.caption(
                        f"Latency: "
                        f"{run.get('latency', 0):.3f}s"
                    )

    st.download_button(
        "Download comparison report",
        json.dumps(report, indent=2),
        file_name=(
            f"{version}_{mutation_id}.json"
        ),
        mime="application/json"
    )


# ==========================================
# REGRESSION HEATMAP
# ==========================================

elif page == "Regression Heatmap":

    st.subheader("Regression Heatmap")

    st.caption(
        "Distinguish newly introduced failures "
        "from existing baseline weaknesses."
    )

    heatmap = build_heatmap_data(version)

    if heatmap.empty:

        st.info("No heatmap data available.")
        st.stop()

    counts = heatmap["state"].value_counts()

    a, b, c, d = st.columns(4)

    a.metric(
        "New regressions",
        int(counts.get("Regression", 0))
    )

    b.metric(
        "Stable passes",
        int(counts.get("Stable pass", 0))
    )

    c.metric(
        "Existing failures",
        int(counts.get("Existing failure", 0))
    )

    d.metric(
        "Improvements",
        int(counts.get("Improvement", 0))
    )

    st.plotly_chart(
        regression_heatmap(heatmap),
        width="stretch"
    )

    st.caption(
        "Each cell represents one mutation-case "
        "comparison, not a unique test case."
    )

    states = st.multiselect(
        "Filter detailed results",
        sorted(
            heatmap["state"].unique()
        ),
        default=sorted(
            heatmap["state"].unique()
        )
    )

    filtered = heatmap[
        heatmap["state"].isin(states)
    ]

    st.dataframe(
        filtered,
        hide_index=True,
        width="stretch"
    )

    st.download_button(
        "Export heatmap data",
        filtered.to_csv(index=False),
        file_name=f"heatmap_{version}.csv",
        mime="text/csv"
    )


# ==========================================
# EFFICIENCY ANALYTICS
# ==========================================

elif page == "Efficiency Analytics":

    st.subheader("Inference efficiency")

    a, b, c, d = st.columns(4)

    a.metric(
        "Uncached calls",
        metrics["uncached_calls"]
    )

    b.metric(
        "Planned calls",
        metrics["planned_calls"]
    )

    c.metric(
        "Saved calls",
        metrics["saved_calls"]
    )

    d.metric(
        "Cache hit rate",
        format_percent(
            metrics["cache_hit_rate"]
        )
    )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            efficiency_chart(metrics),
            width="stretch"
        )

    with right:

        st.plotly_chart(
            duration_chart(df),
            width="stretch"
        )

    st.subheader("Token consumption")

    st.metric(
        "Recorded new inference tokens",
        f"{metrics['new_tokens']:,}"
    )

    st.caption(
        "These are recorded evaluation tokens, "
        "not provider billing totals. "
        "API retries may add unrecorded usage."
    )

    columns = [
        "mutation_id",
        "cache_hit",
        "planned_calls",
        "saved_calls",
        "new_tokens",
        "duration_seconds"
    ]

    st.dataframe(
        df[
            [
                col for col in columns
                if col in df.columns
            ]
        ],
        hide_index=True,
        width="stretch"
    )


# ==========================================
# EXPERIMENT REGISTRY
# ==========================================

elif page == "Experiment Registry":

    st.subheader("Experiment Registry")

    records = []

    for benchmark_version in versions:

        benchmark_df = load_benchmark(
            benchmark_version
        )

        if benchmark_df.empty:
            continue

        result = calculate_metrics(
            benchmark_df
        )

        records.append({

            "Benchmark": benchmark_version,

            "Mutations":
                result["total_mutations"],

            "Detection rate":
                format_percent(
                    result["detection_rate"]
                ),

            "False alarm rate":
                format_percent(
                    result["false_alarm_rate"]
                ),

            "Baseline score":
                format_percent(
                    result["baseline_accuracy"]
                ),

            "Candidate evaluations":
                result["candidate_evaluations"],

            "Saved calls":
                result["saved_calls"],

            "Call reduction":
                format_percent(
                    result["call_reduction"]
                ),

        })

    registry = pd.DataFrame(records)

    st.dataframe(
        registry,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Benchmark versions can use different "
        "test coverage and configurations."
    )

    st.download_button(
        "Export experiment registry",
        registry.to_csv(index=False),
        file_name="experiment_registry.csv",
        mime="text/csv"
    )


    st.divider()
    render_live_registry()


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    PROMPTGATE / LLM RELIABILITY ENGINEERING
    &nbsp; • &nbsp;
    EVALUATION INTELLIGENCE STUDIO
</div>
""", unsafe_allow_html=True)
