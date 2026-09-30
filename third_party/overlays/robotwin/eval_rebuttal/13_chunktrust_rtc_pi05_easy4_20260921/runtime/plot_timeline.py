"""Export a physical-time or native-wall-time timeline from measured events."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    episode=a.episode.parent if a.episode.name=='result.json' else a.episode
    result=json.loads((episode/'result.json').read_text())
    events=[json.loads(x) for x in (episode/'events.jsonl').read_text().splitlines()]
    controlled=result['controlled_clock']
    commits=[x for x in events if x['event']=='commit']
    origin=0 if controlled else min(x['request_wall'] for x in commits)
    dt=result['physics_s']/max(result['physics_ticks'],1)
    fig,ax=plt.subplots(figsize=(10,2.8),layout='constrained')
    for event in events:
        if event['event']=='action':
            start=event['start_tick']*dt if controlled else event['start_wall']-origin
            end=event['end_tick']*dt if controlled else event['end_wall']-origin
            ax.broken_barh([(start,end-start)],(.1,.6),facecolors='#3784bb',edgecolors='white',linewidth=.3)
        if event['event']=='commit':
            start=event['request_tick']*dt if controlled else event['request_wall']-origin
            ready=event['ready_tick']*dt if controlled else event['available_wall']-origin
            commit=event['commit_tick']*dt if controlled else event['commit_wall']-origin
            ax.broken_barh([(start,ready-start)],(1.1,.6),facecolors='#eaa64a',alpha=.85)
            ax.plot([commit,commit],[.05,1.85],color='#3e7152',linewidth=.7)
            ax.text(commit,1.9,f"K={event['selected_k']}",fontsize=6,rotation=60)
    ax.set_yticks([.4,1.4],['TOPP execution','Inference availability'])
    ax.set_xlabel('Simulated physical time (s)' if controlled else 'Measured wall time (s)')
    ax.set_title(f"{result['backbone']} · {result['task']} · {result['method']} · seed {result['episode_seed']}",fontsize=10)
    ax.set_ylim(0,2.7);ax.spines[['top','right']].set_visible(False)
    output=a.output or episode/'timeline.pdf';output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output);fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)

if __name__=='__main__':main()
