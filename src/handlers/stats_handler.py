from typing import Dict, List, Deque
from collections import deque
import wandb


log_stats : Dict = dict()

score_window : Deque = deque(maxlen=100)
completion_window : Deque = deque(maxlen=100)
min_steps_window : Deque = deque(maxlen=100)

action_count : List 
ep_score : float 
min_steps_to_complete : int 


def on_episode_end(ep_id):
    log_stats['episode'] = ep_id
    wandb.log(log_stats)
    log_stats.clear()          #reset dic 



