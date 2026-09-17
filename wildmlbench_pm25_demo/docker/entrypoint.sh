#!/bin/sh
set -eu

if [ ! -f /.dockerenv ]; then
  echo 'This launcher must run inside Docker.' >&2
  exit 1
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo 'Required environment variable: OPENAI_API_KEY' >&2
  exit 2
fi

AIDE_STEPS="${AIDE_STEPS:-3}"
export AIDE_STEPS

python - <<'PY'
from pathlib import Path

workspace = Path('/workspace')
allowed = {'train.csv', 'test_features.csv', 'TASK.md'}
if not workspace.is_dir() or workspace.is_symlink():
    raise SystemExit('Agent workspace is not a real directory.')
entries = list(workspace.iterdir())
if {entry.name for entry in entries} != allowed:
    raise SystemExit(f'Agent workspace allowlist failed: {[entry.name for entry in entries]}')
if any(entry.is_symlink() or not entry.is_file() for entry in entries):
    raise SystemExit('Agent workspace contains a symlink or non-file entry.')
PY

python - <<'PY'
from pathlib import Path
import random
import numpy as np

random.seed(42)
np.random.seed(42)
from aide.interpreter import Interpreter

preflight = Path('/output/preflight')
preflight.mkdir(parents=True, exist_ok=True)
interpreter = Interpreter(preflight, timeout=30)
try:
    result = interpreter.run("print('AIDE_INTERPRETER_OK')")
    if result.exc_type or 'AIDE_INTERPRETER_OK' not in ''.join(result.term_out):
        raise SystemExit('AIDE interpreter preflight failed; paid run was not started.')
finally:
    interpreter.cleanup_session()
PY

python - <<'PY'
import os
import sys
from aide.run import run

steps = os.environ.get('AIDE_STEPS', '3')
if steps is None or not str(steps).strip():
    steps = '3'
try:
    value = int(str(steps).strip())
except ValueError as exc:
    raise SystemExit(f'Invalid AIDE_STEPS value: {steps!r}') from exc
if value <= 0:
    raise SystemExit(f'Invalid AIDE_STEPS value: {steps!r}; expected a positive integer')

sys.argv = [
    'aide',
    'data_dir=/workspace',
    'desc_file=/workspace/TASK.md',
    'log_dir=/output/logs',
    'workspace_dir=/output/workspaces',
    'exp_name=pm25-v1',
    f'agent.steps={value}',
    'agent.search.num_drafts=1',
    'agent.k_fold_validation=1',
    'agent.code.model=o4-mini',
    'agent.feedback.model=gpt-4.1-mini',
    'generate_report=false',
    'copy_data=true',
    'preprocess_data=false',
    'exec.timeout=1200',
]
run()
PY
