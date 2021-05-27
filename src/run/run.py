import wandb
from git import Repo

import tensorflow as tf

from src.run.exec_handler import ExcHandler


def run(episodes : int, par_agent : dict, par_environment : dict, mode : str, checkpoint_file, show : bool, project_path : str, tag : str):

    #check if it's running on gpu 
    physical_devices = tf.config.list_physical_devices('GPU')
    print("GPUs Available:",'True, Num :'+str(len(physical_devices)) if len(physical_devices)>0 else 'False')

    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    #in order to log the current branch from which we are logging to wandb
    repo = Repo(project_path)
    branch_name = repo.active_branch.name

    tags = ['branch_'+branch_name]
    if tag is not None : tags.append(tag)

    config = {
        'num_episodes' : episodes,
        'mode' : mode,
        'from_checkpoint' : checkpoint_file,
        'agent' : par_agent,
        'environment' : par_environment
    }
    wandb.init(config=config, project='flatland-rl', group=mode, tags=tags)
    #set run name to run id 
    wandb.run.name = wandb.run.id

    ex = ExcHandler(par_agent, par_environment, mode, checkpoint_file)
    ex.start(episodes, show)


