from typing import Dict, List, Deque
from collections import deque


id : str = None

#all given parameters 
env_params : Dict = {}
obs_params : Dict = {}
agn_parmas : Dict = {}
trn_params : Dict = {}
training : bool = False 
from_checkpoint : bool = False
num_episodes : int = None

#train stats
max_steps : int = 0
action_size : int = 0
obs_size : int = 0

scores_window : Deque[float] = deque(maxlen=100)  
completion_window : Deque[float] = deque(maxlen=100)
scores : List = []
completion : List = []
action_count : List = [] 


def init(n_episodes, parameters, is_training, checkpoint_file):
    global num_episodes 
    num_episodes= n_episodes
    global training
    training = is_training
    global from_checkpoint
    from_checkpoint = True if checkpoint_file else False
    global env_params
    env_params = parameters['env']
    global obs_params
    obs_params = parameters['obs']
    global agn_parmas
    agn_parmas = parameters['agn']
    global trn_params
    trn_params = parameters['trn']
