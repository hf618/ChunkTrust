"""Versioned panel membership without rewriting original rollout records."""
from functools import lru_cache
import hashlib
import json
from common import EXP


def load_active_protocol():
    pointer=EXP/'artifacts/active_panel.json'
    path=EXP/json.loads(pointer.read_text())['protocol_file'] if pointer.exists() else EXP/'artifacts/formal_protocol.json'
    return json.loads(path.read_text()) if path.exists() else None


def storage_k(protocol, backbone, method):
    # For adaptive methods this field is a historical output-path label only.
    return protocol.get('ahs_record_k',protocol['fixed_k'])[backbone] if method.endswith('ahs') else protocol['fixed_k'][backbone]


def settings_path(protocol, backbone, task, method):
    if 'settings_files' in protocol:
        return EXP/protocol['settings_files'][backbone][task]['ahs' if method.endswith('ahs') else 'fixed']
    return EXP/'artifacts'/f'formal_settings_{backbone}_{task}.json'


@lru_cache(maxsize=64)
def settings_digest(path):
    return hashlib.sha256(json.dumps(json.loads(path.read_text()),sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=32)
def cohort_identity_map(task, cohort):
    path=EXP/'manifests'/cohort/f'{task}.json'
    data=json.loads(path.read_text())
    return hashlib.sha256(path.read_bytes()).hexdigest(), {e['episode_seed']:e for e in data['entries']}


def primary_records(rows, protocol=None):
    protocol=protocol or load_active_protocol()
    if protocol is None:return []
    selected=[];identities=set()
    for row in rows:
        if row['fixed_k']!=storage_k(protocol,row['backbone'],row['method']):continue
        if row['execution_hash']!=protocol['execution_hash']:raise RuntimeError('primary execution hash mismatch')
        path=settings_path(protocol,row['backbone'],row['task'],row['method'])
        if row['settings_hash']!=settings_digest(path):raise RuntimeError('primary settings hash mismatch')
        if protocol.get('formal_n_per_cell',50)==100 and row.get('stage')=='formal':
            cohort='formal' if row['rollout_id']<50 else 'formal_extra50'
            manifest_hash,entries=cohort_identity_map(row['task'],cohort)
            entry=entries.get(row['episode_seed'])
            if (row['manifest_hash']!=manifest_hash or entry is None or
                row['rollout_id']!=entry['rollout_id'] or row['reset_signature']!=entry['signature']):
                raise RuntimeError('primary manifest identity mismatch')
        key=tuple(row[k] for k in ('backbone','task','method','extra_ms','episode_seed'))
        if key in identities:raise RuntimeError('duplicate primary episode identity')
        identities.add(key);selected.append(row)
    return selected
