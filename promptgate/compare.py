
from .runner import summarize


def compare(base, cand, cases, cfg):
    """
    Compare baseline and candidate prompt results.
    Return PASS, FAIL, or ERROR.
    """

    bs = summarize(base, cases)
    cs = summarize(cand, cases)

    group_of = {
        case["id"]: case["group"]
        for case in cases
    }

    critical_group = cfg.get("critical_group", "critical")

    flips_down = []
    flips_up = []

    for case in cases:
        case_id = case["id"]

        baseline_passed = (
            base[case_id]["pass_rate"] >= 0.5
        )

        candidate_passed = (
            cand[case_id]["pass_rate"] >= 0.5
        )

        if baseline_passed and not candidate_passed:
            flips_down.append(case_id)

        if candidate_passed and not baseline_passed:
            flips_up.append(case_id)

    reasons = []

    overall_drop = bs["score"] - cs["score"]

    if overall_drop > cfg["tolerance"]:
        reasons.append(
            f"Overall score dropped {overall_drop:.1%} "
            f"(tolerance {cfg['tolerance']:.1%})"
        )

    for group, baseline_score in bs["groups"].items():
        group_drop = (
            baseline_score - cs["groups"][group]
        )

        if group_drop > cfg["group_tolerance"]:
            reasons.append(
                f"Group '{group}' dropped {group_drop:.1%} "
                f"(tolerance {cfg['group_tolerance']:.1%})"
            )

    critical_regressions = [
        case_id
        for case_id in flips_down
        if group_of[case_id] == critical_group
    ]

    if critical_regressions:
        reasons.append(
            f"{len(critical_regressions)} critical case(s) "
            f"flipped pass to fail: "
            f"{', '.join(critical_regressions)}"
        )

    if bs["errors"] or cs["errors"]:
        verdict = "ERROR"

        reasons.insert(
            0,
            f"API errors: baseline {bs['errors']}, "
            f"candidate {cs['errors']}. "
            f"Result not trustworthy."
        )

    else:
        verdict = "FAIL" if reasons else "PASS"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "flips_down": flips_down,
        "flips_up": flips_up,
        "base": bs,
        "cand": cs,
    }
