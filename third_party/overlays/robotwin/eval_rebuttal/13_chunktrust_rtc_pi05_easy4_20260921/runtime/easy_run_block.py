"""Run the unchanged batch-10 policy/controller under demo_clean."""
import argparse
from pathlib import Path
from common import EXP, write_json
from easy_contract import verify
from panel_run_block import panel_jobs,verify_saved_jobs
from run_block import run
from audit_semantics import audit


def main(args):
    p=verify()
    assert args.backbone=='pi05' and args.task in p['tasks'] and args.stage=='formal'
    assert args.panel_id==p['panel_protocol_id'] and args.fixed_k==40
    groups=panel_jobs(args,p)
    if not verify_saved_jobs(groups):
        print('ALL_EASY_EPISODES_REUSED_WITH_VERIFIED_IDENTITIES',flush=True);return
    import jax
    if jax.default_backend()!='gpu':raise RuntimeError(f'Easy formal inference requires GPU: {jax.devices()}')
    print('EASY_FORMAL_GPU_CONFIRMED',jax.devices(),flush=True)
    from engine import Engine
    engine=Engine('pi05',args.task)
    for jobs in groups:
        # Same first wave as run(engine,jobs); audit its logs before proceeding.
        run(engine,jobs[:10],workers=10)
        proof=audit('formal',paths=[Path(j['output'])/'result.json' for j in jobs[:10]])
        proof.update(setting='demo_clean',task=args.task,method=jobs[0]['method'],extra_ms=jobs[0]['extra_ms'])
        write_json(EXP/'artifacts'/f"easy_wave_audit_{args.task}_{jobs[0]['method']}_{jobs[0]['extra_ms']}.json",proof)
        run(engine,jobs,workers=10)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('backbone','task','stage','panel-id'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--methods',nargs='+',required=True);parser.add_argument('--extra-ms',nargs='+',type=int,default=[0,100,200])
    parser.add_argument('--fixed-k',type=int,default=40)
    main(parser.parse_args())
