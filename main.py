import json
from handler import ExcHandler


def main(parameters_filename, training : bool, rendering : bool, checkpoint_file = None):
    with open(parameters_filename) as json_file:
        parameters = json.load(json_file)

    ex = ExcHandler(parameters, training , rendering, checkpoint_file)
    ex.start(parameters)