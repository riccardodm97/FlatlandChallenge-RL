import json
from argparse import ArgumentParser

import wandb

from src.handlers.exec_handler import ExcHandler


def eval_main(episodes : int, parameter_file, mode : str, checkpoint_file, show : bool): 
    with open(parameter_file) as json_file:
        par = json.load(json_file)
        
    config = {
        'num_episodes' : episodes,
        'mode' : mode,
        'from_checkpoint' : checkpoint_file,
        'agent' : par['agent'],
        'environment' : par['environment']
    }
    wandb.init(config=config, project='flatland-rl', group=mode)
    #set run name to run id 
    wandb.run.name = wandb.run.id
    wandb.run.save()

    ex = ExcHandler(par['agent'], par['environment'], mode, checkpoint_file)
    ex.start(episodes, show)

    #TODO : save stats



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--episodes", dest="episodes", help="Number of episodes to run", default=10, type=int)
    parser.add_argument("-p", "--parameters", dest="parameters", help="Parameter file", default='parameters/eval_par/eval1.json')
    parser.add_argument('-c', '--checkpoint', dest="checkpoint", help="Checkpoint file to be loaded", default=None)
    parser.add_argument('-s', '--show', dest="show", help="Wheter or not render env", action='store_true')
    args = parser.parse_args()


    eval_main(args.episodes, args.parameters, 'eval', args.checkpoint, args.show)

    