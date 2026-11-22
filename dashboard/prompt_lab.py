import difflib
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from promptgate.evaluation import evaluate_prompts, LIVE_RESULTS
from promptgate.runner import load_cases

ROOT = Path(__file__).resolve().parents[1]


def render_result(report, prefix='live'):
    result = report['comparison']
    verdict = result['verdict']
    st.subheader('Evaluation results')
    message = {'PASS': 'PASS — No threshold violated', 'FAIL': 'FAIL — Regression detected', 'ERROR': 'ERROR — Evaluation incomplete'}.get(verdict, verdict)
    {'PASS': st.success, 'FAIL': st.error, 'ERROR': st.warning}.get(verdict, st.info)(message)
    base = result['base']['score']
    cand = result['cand']['score']
    a, b, c = st.columns(3)
    a.metric('Baseline score', f'{base:.1%}')
    b.metric('Candidate score', f'{cand:.1%}')
    c.metric('Score drop', f'{(base-cand)*100:.1f} pp')
    for reason in result['reasons']:
        st.write(reason)
    cache = report['cache']
    a, b, c = st.columns(3)
    a.metric('Baseline reuse', 'HIT' if cache['baseline_cache_hit'] else ('MISS' if cache['enabled'] else 'OFF'))
    b.metric('Planned API calls', cache['planned_api_calls'])
    c.metric('Calls saved', cache['api_calls_saved'])
    if cache.get('warning'):
        st.warning(cache['warning'])
    st.caption('Planned calls exclude retries. Cached scores describe a previous baseline evaluation.')
    st.caption(f"Run: {report['run_id']} | {report['timestamp']}")
    baseline = report['baseline_runs']
    candidate = report['candidate_runs']
    cases = {case['id']: case for case in report['cases']}
    ids = [cid for cid in cases if cid in baseline and cid in candidate]
    rows = []
    for cid in ids:
        x, y = baseline[cid], candidate[cid]
        delta = y['pass_rate'] - x['pass_rate']
        error = any(run.get('error') for run in x['runs'] + y['runs'])
        state = 'Execution error' if error else ('Regression' if delta < 0 else 'Improvement' if delta > 0 else 'Stable pass' if x['pass_rate'] == 1 else 'Existing failure' if x['pass_rate'] == 0 else 'Stable partial pass')
        rows.append({'Case': cid, 'Group': cases[cid]['group'], 'Baseline': f"{x['pass_rate']:.0%}", 'Candidate': f"{y['pass_rate']:.0%}", 'State': state})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if ids:
        st.subheader('Response inspector')
        cid = st.selectbox('Test case', ids, key=f'{prefix}_case_{report["run_id"]}')
        st.write(cases[cid]['input'])
        for col, label, entry in zip(st.columns(2), ['Baseline', 'Candidate'], [baseline[cid], candidate[cid]]):
            with col:
                st.markdown(f'**{label}**')
                for i, run in enumerate(entry['runs'], 1):
                    with st.expander(f'Run {i}', expanded=True):
                        if run.get('error'):
                            st.error(run['error'])
                        elif run.get('passed'):
                            st.success('Assertions passed')
                        else:
                            st.error('Assertions failed')
                        st.code(run.get('text', ''), language='json')
                        for check, passed in run.get('checks', {}).items():
                            st.write(f"{check}: {'PASS' if passed else 'FAIL'}")
                        st.caption(f"Latency: {run.get('latency', 0):.3f}s")
    st.subheader('Export reports')
    folder = LIVE_RESULTS / report['run_id']
    for col, name, mime in zip(st.columns(3), ['comparison.json', 'report.html', 'report.md'], ['application/json', 'text/html', 'text/markdown']):
        path = folder / name
        if path.is_file():
            col.download_button(f'Download {name}', path.read_bytes(), file_name=name, mime=mime, key=f'{prefix}_{name}_{report["run_id"]}')


def render_prompt_lab():
    st.subheader('Live Prompt Lab')
    st.caption('Edit prompts, execute real model evaluations and inspect regressions.')
    manifest = json.loads((ROOT / 'experiments/mutations_manifest.json').read_text(encoding='utf-8'))
    templates = {'Current candidate': ROOT / 'prompts/candidate.txt'}
    templates.update({f"{m['id']} — {m['name']}": ROOT / 'experiments/mutated_prompts' / f"{m['id']}.txt" for m in manifest})
    original = (ROOT / 'prompts/prompt.txt').read_text(encoding='utf-8')
    st.session_state.setdefault('lab_baseline', original)
    st.session_state.setdefault('lab_candidate', templates['Current candidate'].read_text(encoding='utf-8'))
    template = st.selectbox('Experiment template', list(templates), key='lab_template')
    if st.button('Load template'):
        st.session_state.lab_baseline = original
        st.session_state.lab_candidate = templates[template].read_text(encoding='utf-8')
    left, right = st.columns(2)
    baseline = left.text_area('Baseline prompt', key='lab_baseline', height=320)
    candidate = right.text_area('Candidate prompt', key='lab_candidate', height=320)
    diff = '\n'.join(difflib.unified_diff(baseline.splitlines(), candidate.splitlines(), fromfile='baseline', tofile='candidate', lineterm=''))
    with st.expander('Prompt difference', expanded=True):
        if diff:
            st.code(diff, language='diff')
        else:
            st.info('Prompts are identical.')
    cfg = yaml.safe_load((ROOT / 'promptgate.yaml').read_text(encoding='utf-8'))
    cases = load_cases(ROOT / 'testset/cases.jsonl')
    a, b, c = st.columns(3)
    scope = a.selectbox('Test coverage', ['Quick (3 cases)', 'Extended (10 cases)', 'Critical only', 'Full suite'])
    repeats = int(b.number_input('Repeats per case', min_value=1, max_value=3, value=1))
    cache = c.checkbox('Reuse baseline cache', value=True)
    limit = {'Quick (3 cases)': 3, 'Extended (10 cases)': 10}.get(scope)
    critical = scope == 'Critical only'
    selected = [case for case in cases if case['group'] == cfg.get('critical_group', 'critical')] if critical else cases
    if limit is not None:
        selected = selected[:limit]
    calls = len(selected) * repeats
    st.caption(f"Model: {cfg['model']} | {len(selected)} cases | Planned calls: {calls if cache else 2*calls}–{2*calls} (before retries)")
    st.caption('Prompts and responses are sent to the configured provider and saved locally. Use synthetic data only. Large runs may encounter provider rate limits.')
    if st.button('Run evaluation', type='primary', disabled=not (baseline.strip() and candidate.strip() and selected)):
        st.session_state.pop('lab_report', None)
        progress = st.progress(0.0, text='Preparing evaluation...')
        try:
            report = evaluate_prompts(baseline, candidate, limit=limit, repeats=repeats, use_cache=cache, critical_only=critical, progress_callback=lambda value, message: progress.progress(value, text=message))
            st.session_state.lab_report = report
        except Exception as error:
            progress.empty()
            st.error(f'Evaluation could not complete: {error}')
    report = st.session_state.get('lab_report')
    if report:
        if report['baseline_prompt'] != baseline or report['candidate_prompt'] != candidate:
            st.info('Displayed results belong to the previous evaluated prompts, not the current editors.')
        render_result(report)


def render_live_registry():
    st.subheader('Live experiments')
    reports = {}
    for path in sorted(LIVE_RESULTS.glob('*/comparison.json'), reverse=True):
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
            if report['run_id'] == path.parent.name and 'comparison' in report:
                reports[report['run_id']] = report
        except (OSError, ValueError, KeyError):
            continue
    if not reports:
        st.info('No live experiments saved yet. Run an evaluation in Prompt Lab.')
        return
    st.dataframe(pd.DataFrame([{'Run': rid, 'Timestamp': r['timestamp'], 'Verdict': r['comparison']['verdict'], 'Cases': r['configuration']['cases'], 'Repeats': r['configuration']['repeats']} for rid, r in reports.items()]), hide_index=True, width="stretch")
    rid = st.selectbox('Inspect saved experiment', list(reports), key='registry_run')
    with st.expander('Evaluated prompts'):
        a, b = st.columns(2)
        a.code(reports[rid]['baseline_prompt'], language=None)
        b.code(reports[rid]['candidate_prompt'], language=None)
    render_result(reports[rid], prefix='registry')

