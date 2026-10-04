import json
import random
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}

def get_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)

def read_config(path):
    with open(path, encoding='utf-8') as f:
        c = json.load(f)
    if c['train']['batch_size'] < 2:
        raise ValueError('Training batch_size must be >=2 for contrastive loss')
    if c['train']['accumulation'] < 1 or c['train']['epochs'] < 1:
        raise ValueError('epochs and accumulation must be positive')
    if c['loss'].get('temperature', .07) <= 0:
        raise ValueError('temperature must be positive')
    return c

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)
