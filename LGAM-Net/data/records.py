import json
from pathlib import Path

def read_records(path):
    with open(path, encoding='utf-8-sig') as f:
        content = f.read()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = [value] if 'img_local_path' in value else value.get('data', value.get('annotations'))
    if not isinstance(value, list) or not value:
        raise ValueError(f'{path}: expected nonempty JSON array or JSONL')
    return value

def image_path(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f'Image path escapes data root: {relative}')
    return p

def label_of(record):
    label = record.get('context_label', record.get('label'))
    if isinstance(label, str):
        label = {'ooc': 1, 'not-ooc': 0, '0': 0, '1': 1}.get(label.lower(), label)
    if label not in (0, 1):
        raise ValueError('Test requires context_label/label equal to 0 or 1')
    return int(label)
