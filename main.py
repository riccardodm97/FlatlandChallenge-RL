import json
from datetime import datetime
from argparse import ArgumentParser
import wandb



from src.handlers.exec_handler import ExcHandler
import src.handlers.stats_handler as stats


def main(episodes : int, parameters_filename, training : bool, checkpoint_file): 
    with open(parameters_filename) as json_file:
        parameters = json.load(json_file)

    # Unique ID for this run
    now = datetime.now()
    id = now.strftime('%d/%m/%H:%M:%S')
    config = {
        'num_episodes' : episodes,
        'training' : training,
        'from_checkpoint' : True if checkpoint_file else False,
        'env' : parameters['env'],
        'obs' : parameters['obs'],
        'agn' : parameters['agn'],
        'trn' : parameters['trn'],
    }
    wandb.init(config=config,project='flatland-rl',name=id)

    ex = ExcHandler(parameters, training , checkpoint_file)
    ex.start(episodes)

    #TODO : save stats



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--episodes", dest="episodes", help="Number of episodes to run", default=500, type=int)
    parser.add_argument("-p", "--parameters", dest="parameters", help="Parameter file to use", default='parameters/example.json')
    parser.add_argument('-t', '--training', dest="training", help="Enables training", action='store_true')
    parser.add_argument('-c', '--checkpoint', dest="checkpoint", help="Checkpoint file to be loaded", default=None)
    args = parser.parse_args()


    main(args.episodes, args.parameters, args.training, args.checkpoint)

    