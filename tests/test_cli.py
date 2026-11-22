import copy
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from promptgate import cli


def make_results(cases, repeats=1, passed=True, error=None):
    return {
        case['id']: {
            'runs': [
                {'text': '{}', 'passed': passed, 'checks': {},
                 'error': error, 'latency': 0.1, 'cost': 0.0,
                 'input_tokens': 10, 'output_tokens': 5}
                for _ in range(repeats)
            ],
            'pass_rate': float(passed),
        }
        for case in cases
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, 'CACHE_DIR', tmp_path / 'results/cache/baselines')
    monkeypatch.setenv('LLM_BASE_URL', 'https://example.invalid/v1')
    monkeypatch.setenv('LLM_API_KEY', 'test-key')
    Path('prompts').mkdir()
    Path('testset').mkdir()
    Path('prompts/base.txt').write_text('Baseline', encoding='utf-8')
    Path('prompts/cand.txt').write_text('Candidate', encoding='utf-8')
    cases = [
        {'id': 't01', 'group': 'critical', 'input': 'Payment failed',
         'priority': 'high', 'category': 'billing'},
        {'id': 't02', 'group': 'standard', 'input': 'Question',
         'priority': 'low', 'category': 'other'},
    ]
    Path('testset/cases.jsonl').write_text(
        '\n'.join(json.dumps(c) for c in cases), encoding='utf-8')
    cfg = {'model': 'test-model', 'repeats': 1, 'temperature': 0.2,
           'workers': 1, 'tolerance': 0.05, 'group_tolerance': 0.05,
           'critical_group': 'critical'}
    Path('promptgate.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    argv = ['promptgate', '--baseline', 'prompts/base.txt',
            '--candidate', 'prompts/cand.txt']
    monkeypatch.setattr(sys, 'argv', argv)
    def fake_suite(prompt, cases, model, repeats=3, temperature=0, workers=2):
        return make_results(cases, repeats)
    suite = Mock(side_effect=fake_suite)
    monkeypatch.setattr(cli, 'run_suite', suite)
    return cases, cfg, argv, suite


def record():
    return json.loads(Path('results/comparison.json').read_text(encoding='utf-8'))


def test_cache_miss_and_hit(project, capsys):
    cases, cfg, argv, suite = project
    assert cli.main() == 0
    first = record()
    assert first['comparison']['verdict'] == 'PASS'
    assert first['cache']['enabled'] is True
    assert first['cache']['baseline_cache_hit'] is False
    assert first['cache']['planned_api_calls'] == 4
    assert first['cache']['api_calls_saved'] == 0
    assert suite.call_count == 2
    assert 'Baseline cache: MISS' in capsys.readouterr().out
    assert Path('results/report.html').is_file()
    assert Path('results/report.md').is_file()
    assert cli.main() == 0
    second = record()
    assert suite.call_count == 3
    assert second['cache']['baseline_cache_hit'] is True
    assert second['cache']['planned_api_calls'] == 2
    assert second['cache']['api_calls_saved'] == 2
    assert second['cache']['cache_key'] == first['cache']['cache_key']
    assert second['baseline_runs'] == first['baseline_runs']
    assert 'Baseline cache: HIT' in capsys.readouterr().out


def test_disabled_cache(project, capsys):
    cases, cfg, argv, suite = project
    argv.append('--no-cache')
    assert cli.main() == 0
    assert record()['cache'] == {'enabled': False, 'baseline_cache_hit': False,
        'cache_key': None, 'planned_api_calls': 4, 'api_calls_saved': 0}
    assert not cli.CACHE_DIR.exists()
    assert suite.call_count == 2
    assert 'Baseline cache: DISABLED' in capsys.readouterr().out


def test_cache_write_failure(project, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'save_cached_baseline', Mock(side_effect=OSError('disk full')))
    assert cli.main() == 0
    assert 'Could not save cache: disk full' in capsys.readouterr().out


def test_overrides(project):
    cases, cfg, argv, suite = project
    argv.extend(['--limit', '1', '--repeats', '2'])
    assert cli.main() == 0
    assert suite.call_count == 2
    for call in suite.call_args_list:
        assert call.args[1] == cases[:1]
        assert call.args[2] == cfg['model']
        assert call.kwargs == {'repeats': 2, 'temperature': 0.2, 'workers': 1}
    assert len(record()['candidate_runs']['t01']['runs']) == 2


def test_configuration_defaults(project):
    cases, cfg, argv, suite = project
    for key in ['repeats', 'workers', 'temperature']:
        del cfg[key]
    Path('promptgate.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    assert cli.main() == 0
    assert suite.call_args.kwargs == {'repeats': 3, 'temperature': 0.0, 'workers': 2}


@pytest.mark.parametrize('flag,value,message', [
    ('--limit', '0', '--limit must be at least 1'),
    ('--limit', '-1', '--limit must be at least 1'),
    ('--repeats', '0', '--repeats must be at least 1'),
    ('--repeats', '-1', '--repeats must be at least 1'),
])
def test_invalid_arguments(project, capsys, flag, value, message):
    cases, cfg, argv, suite = project
    argv.extend([flag, value])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    suite.assert_not_called()


@pytest.mark.parametrize('kind,message', [('empty', 'No evaluation cases found'),
    ('duplicate', 'Evaluation case IDs must be unique')])
def test_invalid_cases(project, capsys, kind, message):
    cases, cfg, argv, suite = project
    data = [] if kind == 'empty' else [cases[0], cases[0]]
    Path('testset/cases.jsonl').write_text('\n'.join(map(json.dumps, data)), encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    suite.assert_not_called()


@pytest.mark.parametrize('verdict,exit_code', [('PASS', 0), ('FAIL', 1), ('ERROR', 2)])
def test_verdict_exit_codes(project, monkeypatch, capsys, verdict, exit_code):
    metrics = {'score': 1.0, 'cost': 0.0, 'latency': 0.1,
               'groups': {'critical': 1.0, 'standard': 1.0}}
    result = {'verdict': verdict, 'base': metrics, 'cand': metrics,
              'reasons': ['Test reason'] if verdict != 'PASS' else [],
              'flips_down': [], 'flips_up': []}
    monkeypatch.setattr(cli, 'compare', Mock(return_value=result))
    assert cli.main() == exit_code
    assert record()['comparison']['verdict'] == verdict
    assert f'PROMPTGATE VERDICT: {verdict}' in capsys.readouterr().out


@pytest.mark.parametrize('target', ['result', 'runs', 'run'])
def test_invalid_cache_nested_types(project, target):
    cases = project[0]
    results = make_results(cases)
    if target == 'result':
        results['t01'] = None
    elif target == 'runs':
        results['t01']['runs'] = 'invalid'
    else:
        results['t01']['runs'][0] = None
    assert cli.valid_baseline(results, cases, 1) is False


def test_module_entrypoint(monkeypatch, capsys):
    # --help exits before accessing files or making API requests.
    monkeypatch.setattr(sys, 'argv', ['promptgate', '--help'])
    # Avoid the runpy warning from an already imported module.
    monkeypatch.delitem(sys.modules, 'promptgate.cli', raising=False)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module('promptgate.cli', run_name='__main__')
    assert exc.value.code == 0
    assert 'usage:' in capsys.readouterr().out
