from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
PROTOCOL_ID = 'chunktrust-rtc-v2-fft50-ready-boundary'
TASKS = ('place_a2b_left', 'place_bread_basket', 'place_bread_skillet',
         'place_can_basket', 'blocks_ranking_rgb', 'handover_block',
         'handover_mic', 'hanging_mug')
CAL_TASKS = ('place_a2b_left', 'place_bread_basket', 'handover_block', 'hanging_mug')
METHODS = ('sync_fixed', 'sync_ahs', 'rtc_fixed', 'rtc_ahs')
CONFIGS = {'pi0': ('pi0', 'pi0_base_aloha_robotwin_lora', '30000'),
           'pi05': ('pi05_horizon', 'pi05_base_aloha_robotwin_full', '20000')}

def checkpoint(backbone, task):
    folder, recipe, step = CONFIGS[backbone]
    return ROOT / 'policy' / folder / 'checkpoints' / recipe / f'{backbone}_{task}_clean50' / step

def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x

def write_json(path, obj, immutable=False):
    path = Path(path)
    payload = json.dumps(jsonable(obj), indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text() != payload:
            raise RuntimeError(f'refusing to change frozen artifact {path}')
        return
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(payload)
    temp.replace(path)

def append_json(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(jsonable(obj), allow_nan=False, sort_keys=True) + '\n')
        f.flush()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def rng_seed(backbone, task, seed, request, stream='model'):
    key = f'chunktrust-rtc-v1/{backbone}/{task}/{seed}/0/{request}/{stream}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')

def soft_mask(frozen, overlap, horizon=50):
    if not 0 <= frozen <= overlap <= horizon:
        raise ValueError((frozen, overlap, horizon))
    i = np.arange(horizon)
    c = np.clip((overlap - i) / (overlap - frozen + 1), 0., 1.)
    return np.where(i < frozen, 1., np.where(i < overlap, c * np.expm1(c) / np.expm1(1.), 0.)).astype(np.float32)

def observation_input(obs, prompt):
    return {'state': np.asarray(obs['joint_action']['vector'], dtype=np.float32),
            'images': {k: np.ascontiguousarray(obs['observation'][camera]['rgb'].transpose(2, 0, 1))
                       for k, camera in [('cam_high','head_camera'), ('cam_left_wrist','left_camera'),
                                         ('cam_right_wrist','right_camera')]}, 'prompt': prompt}

def observation_hash(obs):
    return {'state_sha256': digest(np.asarray(obs['state']).tobytes()),
            'images_sha256': {k: digest(v.tobytes()) for k,v in obs['images'].items()},
            'instruction_sha256': digest(obs['prompt'].encode())}
