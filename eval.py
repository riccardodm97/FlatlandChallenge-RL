import json
from argparse import ArgumentParser
import os

from src.run.run import run


def eval_main(episodes : int, parameter_file, mode : str, checkpoint_file, show : bool, tag): 
    with open(parameter_file) as json_file:
        par = json.load(json_file)

    par_agent = par['agent']
    par_environment = par['environment']

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        
    run(episodes,par_agent,par_environment,mode,checkpoint_file,show,PROJECT_ROOT,tag)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--episodes", dest="episodes", help="Number of episodes to run", default=10, type=int)
    parser.add_argument("-p", "--parameters", dest="parameters", help="Parameter file", default='parameters/eval_par/eval1.json')
    parser.add_argument('-c', '--checkpoint', dest="checkpoint", help="Checkpoint file to be loaded", default=None)
    parser.add_argument('-s', '--show', dest="show", help="Wheter or not render env", action='store_true')
    parser.add_argument("-t",  "--tag", dest="tag", help="Useful to tag a run in wandb", default=None)

    args = parser.parse_args()


    eval_main(args.episodes, args.parameters, 'eval', args.checkpoint, args.show, args.tag)

    