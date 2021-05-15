import json
from datetime import datetime
from argparse import ArgumentParser
import wandb

from src.handlers.exec_handler import ExcHandler


def eval_main(episodes : int, parameter_file, mode : str, checkpoint_file, show : bool): 
    with open(parameter_file) as json_file:
        par = json.load(json_file)
        
    # Unique ID for this run
    now = datetime.now()
    id = now.strftime('%d/%m/%H:%M:%S')
    config = {
        'num_episodes' : episodes,
        'mode' : mode,
        'from_checkpoint' : checkpoint_file,
        'par' : par
    }
    wandb.init(config=config, project='flatland-rl', name=id, group=mode, monitor_gym=True)

    ex = ExcHandler(par['agent'], par['environment'], mode, checkpoint_file)
    ex.start(episodes, show)

    #TODO : save stats



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--episodes", dest="episodes", help="Number of episodes to run", default=10, type=int)
    parser.add_argument("-p", "--parameters", dest="parameters", help="Parameter file", default='parameters/eval_par/eval1.json')
    parser.add_argument('-c', '--checkpoint', dest="checkpoint", help="Checkpoint file to be loaded", default=None)
    parser.add_argument('-s', '--show', dest="show", help="Wheter or not render env", default=False)
    args = parser.parse_args()


    eval_main(args.episodes, args.parameters,'eval', args.checkpoint, args.show)

    