import json
from argparse import ArgumentParser
import os

from src.run.run import run


def train_main(episodes : int, agn_par_file, env_par_file, mode : str, checkpoint_file, tag : str ): 

    agn_par_file = 'parameters/agn_train_par/'+agn_par_file
    env_par_file = 'parameters/env_train_par/'+env_par_file
    with open(agn_par_file) as agn_json_file, open(env_par_file) as env_json_file:
        par_agent = json.load(agn_json_file)
        par_environment = json.load(env_json_file)
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    run(episodes,par_agent,par_environment,mode,checkpoint_file,False,PROJECT_ROOT,tag)
     
 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e",  "--episodes", dest="episodes", help="Number of episodes to run", default=500, type=int)
    parser.add_argument("-pa", "--par_agn", dest="par_agn", help="Agent parameter file", default='agn_base.json')
    parser.add_argument("-pe", "--par_env", dest="par_env", help="Env parameter file", default='env_base.json')
    parser.add_argument("-c",  "--checkpoint", dest="checkpoint", help="Checkpoint file to be loaded", default=None)
    parser.add_argument("-t",  "--tag", dest="tag", help="Useful to tag a run in wandb", default=None)
    args = parser.parse_args()


    train_main(args.episodes, args.par_agn, args.par_env, 'train', args.checkpoint, args.tag)

    