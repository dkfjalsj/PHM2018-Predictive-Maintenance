"""Local configuration. Environment variables override the ignored .env file."""
from pathlib import Path
from datetime import datetime, timezone
import os
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def read_env(path=None):
    path = Path(path) if path is not None else PROJECT_ROOT / '.env'
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if not sep or not key.strip().replace('_', '').isalnum():
            raise ValueError('Invalid local environment configuration line')
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result

def setting(key):
    return os.environ.get(key, read_env().get(key, ''))

def data_root():
    value = setting('PHM2018_DATA_ROOT')
    if not value:
        raise ValueError('Set PHM2018_DATA_ROOT in .env or the environment')
    path = Path(value).expanduser().resolve()
    for sub in ['train', 'train/train_faults', 'train/train_ttf']:
        if not (path / sub).is_dir():
            raise ValueError('Dataset root is missing required training subdirectories')
    return path

def output_dir(stage):
    base = setting('PHM2018_OUTPUT_ROOT')
    base = Path(base) if base else PROJECT_ROOT / 'reports' / 'generated'
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    run_id = setting('PHM2018_RUN_ID') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    if Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('PHM2018_RUN_ID must be a simple directory name')
    path = base.resolve() / run_id / stage
    # Refuse every overwrite, including an explicitly repeated run ID.
    path.mkdir(parents=True, exist_ok=False)
    return path

def report_workdir():
    value = setting('PHM2018_SENSOR_REPORT_DIR')
    if not value:
        raise ValueError('Set PHM2018_SENSOR_REPORT_DIR to a disposable copy or a new validation output directory')
    path = Path(value)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    path = path.resolve()
    protected = [PROJECT_ROOT/'reports'/'sensor_validation', PROJECT_ROOT/'reports'/'archive']
    if any(path == p.resolve() or p.resolve() in path.parents for p in protected):
        raise ValueError('Preserved report folders cannot be rebuilt in place')
    if not path.is_dir():
        raise ValueError('Report working directory must already exist')
    return path
