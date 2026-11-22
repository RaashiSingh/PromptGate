
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==========================================
# COLOR SYSTEM
# ==========================================

COLORS = {
    "purple": "#8B5CF6",
    "lavender": "#A78BFA",
    "green": "#34D399",
    "red": "#F87171",
    "amber": "#FBBF24",
    "slate": "#64748B",
    "background": "#111827",
    "text": "#E2E8F0",
    "grid": "#263044",
}

OUTCOME_COLORS = {
    "Detected regression": "#8B5CF6",
    "Accepted harmless change": "#34D399",
    "Missed regression": "#F87171",
    "False alarm": "#FBBF24",
    "Execution error": "#64748B",
}


# ==========================================
# COMMON CHART STYLING
# ==========================================

def style_chart(fig, height=350):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(
            family="Arial",
            color=COLORS["text"],
            size=12
        ),
        height=height,
        margin=dict(
            l=25,
            r=25,
            t=55,
            b=25
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_color="#FFFFFF"
        ),
    )

    fig.update_xaxes(
        gridcolor=COLORS["grid"]
    )

    fig.update_yaxes(
        gridcolor=COLORS["grid"]
    )

    return fig


def format_percent(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.1%}"


# ==========================================
# KPI CARDS
# ==========================================

def render_kpis(metrics):

    cols = st.columns(4)

    cols[0].metric(
        "Regression detection",
        format_percent(
            metrics["detection_rate"]
        )
    )

    cols[1].metric(
        "False alarm rate",
        format_percent(
            metrics["false_alarm_rate"]
        )
    )

    cols[2].metric(
        "Completed mutations",
        (
            f"{metrics['completed_mutations']} / "
            f"{metrics['total_mutations']}"
        )
    )

    cols[3].metric(
        "Execution errors",
        metrics["execution_errors"]
    )

    cols = st.columns(4)

    cols[0].metric(
        "Mean baseline score",
        format_percent(
            metrics["baseline_accuracy"]
        )
    )

    cols[1].metric(
        "API calls saved",
        f"{metrics['saved_calls']:,}"
    )

    cols[2].metric(
        "Call reduction",
        format_percent(
            metrics["call_reduction"]
        )
    )

    cols[3].metric(
        "Cache hit rate",
        format_percent(
            metrics["cache_hit_rate"]
        )
    )


# ==========================================
# MUTATION OUTCOME DONUT
# ==========================================

def outcome_chart(df):

    counts = df["outcome"].value_counts()

    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.75,
            marker=dict(
                colors=[
                    OUTCOME_COLORS.get(
                        label,
                        COLORS["slate"]
                    )
                    for label in counts.index
                ],
                line=dict(
                    color=COLORS["background"],
                    width=4
                )
            ),
            textinfo="none",
            hovertemplate=(
                "%{label}<br>"
                "Mutations: %{value}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            )
        )
    )

    fig.add_annotation(
        text=(
            f"<b>{len(df)}</b><br>"
            "<span style='font-size:12px'>"
            "MUTATIONS</span>"
        ),
        x=0.5,
        y=0.5,
        font=dict(size=26),
        showarrow=False
    )

    fig.update_layout(
        title="Mutation outcomes",
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center"
        )
    )

    return style_chart(fig, 370)


# ==========================================
# BASELINE VS CANDIDATE CHART
# ==========================================

def mutation_impact_chart(df):

    data = df.dropna(
        subset=[
            "baseline_score",
            "candidate_score"
        ]
    ).copy()

    labels = (
        data["mutation_id"]
        + " | "
        + data["name"]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Baseline",
            y=labels,
            x=data["baseline_score"] * 100,
            orientation="h",
            marker_color="#475569",
            hovertemplate=(
                "%{y}<br>"
                "Baseline: %{x:.1f}%"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Bar(
            name="Candidate",
            y=labels,
            x=data["candidate_score"] * 100,
            orientation="h",
            marker_color=[
                OUTCOME_COLORS.get(
                    outcome,
                    COLORS["slate"]
                )
                for outcome in data["outcome"]
            ],
            hovertemplate=(
                "%{y}<br>"
                "Candidate: %{x:.1f}%"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Baseline vs candidate performance",
        barmode="group",
        bargap=0.25,
        xaxis=dict(
            title="Test pass rate (%)",
            range=[0, 105]
        ),
        yaxis=dict(
            autorange="reversed"
        ),
        legend=dict(
            orientation="h",
            y=1.08
        )
    )

    return style_chart(
        fig,
        max(450, len(data) * 40)
    )


# ==========================================
# INFERENCE EFFICIENCY CHART
# ==========================================

def efficiency_chart(metrics):

    uncached = metrics["uncached_calls"]
    planned = metrics["planned_calls"]

    fig = go.Figure(
        go.Bar(
            x=[
                "Without cache",
                "With current configuration"
            ],
            y=[
                uncached,
                planned
            ],
            marker_color=[
                "#475569",
                COLORS["purple"]
            ],
            text=[
                uncached,
                planned
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>"
                "Planned calls: %{y}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Inference efficiency",
        yaxis=dict(
            title="Planned API calls",
            range=[
                0,
                max(uncached, planned, 1) * 1.25
            ]
        ),
        showlegend=False
    )

    return style_chart(fig, 370)


# ==========================================
# REGRESSION HEATMAP
# ==========================================

def regression_heatmap(data):

    states = [
        "Existing failure",
        "Stable pass",
        "Regression",
        "Improvement",
        "Execution error",
        "Stable partial pass"
    ]

    state_colors = [
        "#475569",
        "#059669",
        "#EF4444",
        "#22D3EE",
        "#F59E0B",
        "#8B5CF6"
    ]

    state_symbols = {
        "Existing failure": "—",
        "Stable pass": "✓",
        "Regression": "↓",
        "Improvement": "↑",
        "Execution error": "!",
        "Stable partial pass": "~"
    }

    matrix = data.pivot(
        index="mutation_id",
        columns="case_id",
        values="state"
    ).sort_index()

    matrix = matrix.reindex(
        sorted(matrix.columns),
        axis=1
    )

    state_codes = {
        state: index
        for index, state in enumerate(states)
    }

    numeric = [
        [
            state_codes.get(value)
            for value in row
        ]
        for row in matrix.values
    ]

    symbols = [
        [
            state_symbols.get(value, "")
            for value in row
        ]
        for row in matrix.values
    ]

    colorscale = []

    for index, color in enumerate(state_colors):

        colorscale.append([
            index / len(states),
            color
        ])

        colorscale.append([
            (index + 1) / len(states),
            color
        ])

    fig = go.Figure(
        go.Heatmap(
            z=numeric,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            zmin=-0.5,
            zmax=len(states) - 0.5,
            colorscale=colorscale,
            text=symbols,
            texttemplate="%{text}",
            textfont=dict(
                size=15,
                color="white"
            ),
            customdata=matrix.values.tolist(),
            xgap=4,
            ygap=4,
            hoverongaps=False,
            hovertemplate=(
                "Mutation: %{y}<br>"
                "Test case: %{x}<br>"
                "Outcome: %{customdata}"
                "<extra></extra>"
            ),
            showscale=True,
            colorbar=dict(
                tickvals=list(
                    range(len(states))
                ),
                ticktext=states,
                thickness=12,
                len=0.85
            )
        )
    )

    fig.update_layout(
        title="Test-level regression matrix",
        xaxis=dict(
            title="Test cases",
            side="top"
        ),
        yaxis=dict(
            title="Mutations",
            autorange="reversed"
        )
    )

    return style_chart(
        fig,
        max(450, len(matrix) * 34)
    )


# ==========================================
# EXECUTION DURATION CHART
# ==========================================

def duration_chart(df):

    fig = go.Figure(
        go.Bar(
            x=df["mutation_id"],
            y=df["duration_seconds"],
            marker_color=COLORS["purple"],
            hovertemplate=(
                "Mutation: %{x}<br>"
                "Duration: %{y:.2f}s"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Mutation execution duration",
        xaxis_title="Mutation",
        yaxis_title="Seconds"
    )

    return style_chart(fig, 330)
