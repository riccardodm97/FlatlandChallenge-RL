import json
from src.exec import ExcHandler
from argparse import ArgumentParser


def main(episodes : int, parameters_filename, training : bool, checkpoint_file): 
    with open(parameters_filename) as json_file:
        parameters = json.load(json_file)

    ex = ExcHandler(parameters, training , checkpoint_file)
    ex.start(episodes)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-E", "--episodes", dest="episodes", help="Number of episodes to run", default=500, type=int)
    parser.add_argument("-P", "--parameters", dest="parameters", help="Parameter file to use", default='parameters/example.json')
    parser.add_argument('-T', '--training', dest="training", help="Enables training", action='store_true')
    parser.add_argument('-C', '--checkpoint', dest="checkpoint", help="Checkpoint file to be loaded", default=None)
    args = parser.parse_args()


    main(args.episodes, args.parameters, args.training, args.checkpoint)

    