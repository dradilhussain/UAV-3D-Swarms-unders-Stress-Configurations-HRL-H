"""Record the seed-0 episodes of all methods (paths, tasks, obstacles).
NOT needed to re-draw the figures (the notebooks read the stored data in ../data/).
It documents how data/episodes_seed0.pkl was produced from the original project (uav_swarm_3d-v3):
copy it to the project root and run it there, e.g.  python record_episodes.py
"""
import sys, pickle, numpy as np
sys.path[:0]=['code','code/env_3d','code/mappo_3d']
import run_full_eval as R
actor, critic, log, ne, maddpg, qmix = R.train_or_load()
policies, abl = R.build_policies(actor, maddpg, qmix)
out={}
for eid in R.ENV_ORDER:
    out[eid]={'pol':{}}
    for name in ['random','greedy','cbba','mappo','maddpg','qmix','dmpc','hrlh']:
        env,traj,soc=R.record_episode(policies[name],R.ENV_CFGS[eid],seed=0)
        out[eid]['task_pos']=np.array(env.task_pos); out[eid]['base']=np.array(env.base_pos)
        out[eid]['obstacles']=np.array(env.obstacles) if len(env.obstacles) else np.zeros((0,6))
        out[eid]['pol'][name]=dict(traj=np.array(traj),completed=np.array(env.task_completed),
            soc=np.array(soc),stranded=np.array(env.stranded),steps=len(traj)-1)
    print(eid,'ok',flush=True)
pickle.dump(out,open('/home/claude/build/episodes.pkl','wb'))
