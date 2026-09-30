"""Validate the active panel, then use the unchanged rollout implementation."""
import argparse
import json
from pathlib import Path
from common import EXP, METHODS
from panel_config import load_active_protocol, settings_path, storage_k
from run_block import jobs_for, run, execution_fingerprint
from semantic_gate import verify_gate


def panel_jobs(args, protocol):
    if args.stage not in ('formal','strong_fixed','native'):raise ValueError(args.stage)
    if args.stage=='formal' and protocol.get('formal_n_per_cell',50)==100:
        from formal_cohorts import write_combined
        write_combined(args.task)
    if args.stage=='strong_fixed':
        assert args.methods==['rtc_fixed']
        selection=json.loads((EXP/protocol['supplementary_selection_file']).read_text())
        assert selection['panel_protocol_id']==protocol['panel_protocol_id']
        assert args.fixed_k in selection['strong_fixed_k'][args.backbone]
    else:
        assert args.fixed_k==protocol['fixed_k'][args.backbone]
    grouped={}
    for method in args.methods:
        assert method in METHODS
        path=settings_path(protocol,args.backbone,args.task,method)
        settings=json.loads(path.read_text())
        local=argparse.Namespace(**vars(args))
        local.methods=[method]
        local.fixed_k=storage_k(protocol,args.backbone,method) if args.stage!='strong_fixed' else args.fixed_k
        method_jobs=jobs_for(local,settings)
        if args.stage=='formal' and protocol.get('formal_n_per_cell',50)>50:
            assert protocol['formal_n_per_cell']==100
            extension=argparse.Namespace(**vars(local));extension.stage='formal_extra50'
            added=jobs_for(extension,settings)
            for job in added:
                assert 50<=job['entry']['rollout_id']<100
                relative=Path(job['output']).relative_to(EXP/'results/formal_extra50')
                job.update(stage='formal',output=str(EXP/'results/formal'/relative),manifest_cohort='formal_extra50')
            method_jobs+=added
        for job in method_jobs:
            grouped.setdefault((job['extra_ms'],method),[]).append(job)
    return [grouped[(delay,method)] for delay in args.extra_ms for method in args.methods]


def verify_saved_jobs(groups):
    missing=0
    for jobs in groups:
        for job in jobs:
            output=Path(job['output']);path=output/'result.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['status']=='completed'
                for key in ('stage','backbone','task','method','delay_s','fixed_k','candidates','timeout_s','execution_hash','manifest_hash','settings_hash'):
                    if old[key]!=job[key]:raise RuntimeError(f'reused record differs: {key}: {path}')
                assert old['episode_seed']==job['entry']['episode_seed']
                assert old['reset_signature']==job['entry']['signature']
            else:
                if output.exists() and any(output.iterdir()):raise RuntimeError(f'archive interrupted attempt before retry: {output}')
                missing+=1
    return missing


def main(args):
    protocol=load_active_protocol()
    assert protocol['panel_protocol_id']==args.panel_id
    assert protocol['execution_hash']==execution_fingerprint()
    for stage in ('native_check','smoke','pilot'):verify_gate(stage)
    groups=panel_jobs(args,protocol)
    if not verify_saved_jobs(groups):
        print('ALL_EPISODES_REUSED_WITH_VERIFIED_IDENTITIES',flush=True);return
    import jax
    if jax.default_backend() != 'gpu':
        raise RuntimeError(f'Formal model inference requires GPU; found {jax.devices()}')
    print('FORMAL_MODEL_GPU_CONFIRMED',jax.devices(),flush=True)
    from engine import Engine
    engine=Engine(args.backbone,args.task)
    for jobs in groups:run(engine,jobs,workers=1 if args.stage=='native' else 10)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('backbone','task','stage','panel-id'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--methods',nargs='+',required=True)
    parser.add_argument('--extra-ms',nargs='+',type=int,default=[0])
    parser.add_argument('--fixed-k',type=int,required=True)
    main(parser.parse_args())
