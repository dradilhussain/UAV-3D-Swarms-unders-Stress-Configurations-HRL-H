"""Run all 8 methods for seeds 0-9 in the 4 base environments and store every episode result.
NOT needed to re-draw the figures (the notebooks read the stored data in ../data/).
It documents how data/perseed_base.json was produced from the original project (uav_swarm_3d-v3):
copy it to the project root and run it there, e.g.  python perseed_runs.py
"""
import sys, json, numpy as np, pickle
sys.path[:0]=['code','code/env_3d','code/mappo_3d']
import run_full_eval as R
actor, critic, log, ne, maddpg, qmix = R.train_or_load()
policies, abl = R.build_policies(actor, maddpg, qmix)
out={}
for eid in R.ENV_ORDER:
    out[eid]={}
    for name in ['random','greedy','cbba','mappo','maddpg','qmix','dmpc','hrlh']:
        rs=[R.run_episode(policies[name],R.ENV_CFGS[eid],s) for s in range(10)]
        out[eid][name]=[{k:(v if not isinstance(v,(np.floating,np.integer)) else float(v)) for k,v in r.items() if k!='completion_steps'} for r in rs]
    print(eid,'done',flush=True)
json.dump(out,open('/home/claude/perseed.json','w'))
