from typing import Dict, Deque
from collections import deque
import wandb

#stats to be logged through wandb 
log_stats : Dict = dict()

#stats to be saved during a run to be able tu perform further calculations (NOT logged to wandb)
utils_stats : Dict = dict() 

#episode stats 
score_window : Deque = deque(maxlen=100)
completion_window : Deque = deque(maxlen=100)
steps_first_window : Deque = deque(maxlen=100)
steps_last_window : Deque = deque(maxlen=100)
exploration_window : Deque = deque(maxlen=100)
act_timer_window : Deque = deque(maxlen=100)
learn_timer_window : Deque = deque(maxlen=100)


#actual log call to wandb cloud 
def on_episode_end(ep_id):
    log_stats['episode'] = ep_id
    wandb.log(log_stats)

    #reset some values 
    log_stats.clear()




