"""Run one isolated AIDE experiment and evaluate it on the host."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluate_agent_output import evaluate, validate_predictions
from prepare_aide_workspace import prepare, validate_workspace


BASELINE_RMSE = 2.370575


def run_command(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be a positive integer')
    root = Path(__file__).resolve().parents[1]
    workspace = root / 'aide_task'
    runs = root / 'runs'
    run_id = args.run_id or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'agent_name': 'WecoAI AIDE ML',
        'aide_version': '0.2.2',
        'llm_provider': 'OpenAI',
        'model': 'o4-mini / gpt-4.1-mini',
        'agent_steps': args.steps,
        'human_baseline_rmse': BASELINE_RMSE,
        'exit_status': 'not_started',
    }
    (run_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

    prepare(root, workspace)
    validate_workspace(workspace)
    if not os.environ.get('OPENAI_API_KEY'):
        metadata['exit_status'] = 'blocked_missing_OPENAI_API_KEY'
        (run_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        print('Required environment variable: OPENAI_API_KEY', file=sys.stderr)
        return 2

    image = 'wildmlbench-pm25-aide:0.2.2'
    build = run_command(['docker', 'build', '-t', image, str(root / 'docker')], capture_output=True)
    (run_dir / 'docker_build.log').write_text((build.stdout or '') + (build.stderr or ''), encoding='utf-8')
    if build.returncode:
        metadata['exit_status'] = 'docker_build_failed'
        (run_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        return build.returncode

    start = time.monotonic()
    container = run_command([
        'docker', 'run', '--rm', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--pids-limit', '256', '--cpus', '4', '--memory', '8g',
        '--env', 'OPENAI_API_KEY', '--env', f'AIDE_STEPS={args.steps}',
        '--mount', f'type=bind,source={workspace},target=/workspace,readonly',
        '--mount', f'type=bind,source={run_dir},target=/output', image,
    ], capture_output=True)
    (run_dir / 'aide_stdout.log').write_text(container.stdout or '', encoding='utf-8')
    (run_dir / 'aide_stderr.log').write_text(container.stderr or '', encoding='utf-8')
    metadata['runtime_seconds'] = round(time.monotonic() - start, 3)
    metadata['exit_status'] = 'completed' if container.returncode == 0 else f'container_failed_{container.returncode}'

    candidates = list(run_dir.rglob('predictions.csv')) + list(run_dir.rglob('submission.csv'))
    candidates = [path for path in candidates if path.name != 'predictions.csv' or path != run_dir / 'predictions.csv']
    if container.returncode == 0 and candidates:
        selected = max(candidates, key=lambda path: path.stat().st_mtime)
        shutil.copy2(selected, run_dir / 'predictions.csv')
        validation = validate_predictions(run_dir / 'predictions.csv', workspace / 'test_features.csv')
        validation['test_rmse'] = evaluate(run_dir / 'predictions.csv', root / 'data' / 'processed' / 'test_labels.csv')
        validation['absolute_difference'] = abs(validation['test_rmse'] - BASELINE_RMSE)
        validation['percentage_difference'] = (validation['test_rmse'] - BASELINE_RMSE) / BASELINE_RMSE * 100
        (run_dir / 'evaluation.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
        metadata['test_rmse'] = validation['test_rmse']
    elif container.returncode == 0:
        metadata['exit_status'] = 'completed_without_prediction_file'
    (run_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return container.returncode


if __name__ == '__main__':
    raise SystemExit(main())
