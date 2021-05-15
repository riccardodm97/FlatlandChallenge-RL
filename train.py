import json
from datetime import datetime
from argparse import ArgumentParser
import wandb

from src.handlers.exec_handler import ExcHandler


def train_main(episodes : int, agn_par_file, env_par_file, mode : str, checkpoint_file = None): 
    with open(agn_par_file) as agn_json_file, open(env_par_file) as env_json_file:
        agn_par = json.load(agn_json_file)
        env_par = json.load(env_json_file)

    # Unique ID for this run
    now = datetime.now()
    id = now.strftime('%d/%m/%H:%M:%S')
    config = {
        'num_episodes' : episodes,
        'mode' : mode,
        'from_checkpoint' : checkpoint_file,
        'agn' : agn_par,
        'env' : env_par,
    }
    wandb.init(config=config, project='flatland-rl', name=id, group=mode)

    ex = ExcHandler(agn_par, env_par, mode, checkpoint_file)
    ex.start(episodes)

    #TODO : save stats



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--episodes", dest="episodes", help="Number of episodes to run", default=500, type=int)
    parser.add_argument("-pa", "--par_agn", dest="par_agn", help="Agent parameter file", default='parameters/agn_par/agn_base.json')
    parser.add_argument("-pe", "--par_env", dest="par_env", help="Env parameter file", default='parameters/agn_par/env_base.json')
    parser.add_argument('-c', '--checkpoint', dest="checkpoint", help="Checkpoint file to be loaded", default=None)
    args = parser.parse_args()


    train_main(args.episodes, args.par_agn, args.env_par,'train', args.checkpoint)

    