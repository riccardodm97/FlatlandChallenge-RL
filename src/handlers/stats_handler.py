from typing import Dict, List, Deque
from collections import deque
import wandb


episode_stats : Dict = dict()

action_count : List 
ep_score : float 
min_steps_to_complete : int 


def on_episode_end(ep_id):
    episode_stats['episode'] = ep_id
    wandb.log(episode_stats)

