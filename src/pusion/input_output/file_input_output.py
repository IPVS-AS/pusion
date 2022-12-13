import json
import ntpath
import shutil
from pathlib import Path

import pickle


def load_pickle_files_as_data(file_paths):
    """
    Load pickle files containing decision outputs as an data array.

    :param file_paths: A List of file paths to the individual pickle files.
    :return: A data array.
    """
    data = []
    for file_path in file_paths:
        with (open(file_path, "rb")) as handle:
            data.append(pickle5.load(handle))
    return data


def dump_pusion_data(data, file_path):
    """
    Dump classification output data to the given file using pickle.

    :param data: A data dictionary.
    :param file_path: Location of the output pickle file.
    """
    with open(file_path, "wb") as handle:
        pickle5.dump(data, handle, protocol=pickle5.HIGHEST_PROTOCOL)


def dump_data_as_txt(data, name, identifier):
    """
    Dump a data dictionary to the JSON file for a given evaluation unit.

    :param data: A data dictionary.
    :param name: The file name.
    :param identifier: The identifier of the current evaluation unit (e.g. date/time).
    """
    directory = "res/eval_" + identifier
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(directory + "/" + name + ".txt", 'w') as file:
        file.write(json.dumps(data, indent=4))


def save(plot_instance, name, identifier):
    """
    Save the plot instance for a given evaluation unit to the SVG and the PDF file, respectively.

    :param plot_instance: `matplotlib.pyplot`-instance.
    :param name: The file name.
    :param identifier: The identifier of the current evaluation unit (e.g. date/time).
    """
    directory = "res/eval_" + identifier
    Path(directory).mkdir(parents=True, exist_ok=True)
    plot_instance.savefig(directory + "/" + name + ".svg", bbox_inches="tight")
    plot_instance.savefig(directory + "/" + name + ".pdf", bbox_inches="tight")


def save_evaluator(file, identifier):
    """
    Save the evaluation script for a given evaluation unit.

    :param file: The Python file. (E.g. referenced by __file__).
    :param identifier: The identifier of the current evaluation unit (e.g. date/time).
    """
    directory = "res/eval_" + identifier
    Path(directory).mkdir(parents=True, exist_ok=True)
    shutil.copy(file, directory + "/" + ntpath.basename(file) + ".txt")
