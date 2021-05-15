import pickle5
import shutil
from pathlib import Path
import ntpath


def load_native_files_as_data(file_paths):
    data = []
    for file_path in file_paths:
        with (open(file_path, "rb")) as handle:
            data.append(pickle5.load(handle))
    return data


def dump_pusion_data(data, file_path='fusion_output.pickle5'):
    with open(file_path, "wb") as handle:
        pickle5.dump(data, handle, protocol=pickle5.HIGHEST_PROTOCOL)


def save(plot_instance, name, identifier):
    directory = "figs/eval_" + identifier
    Path(directory).mkdir(parents=True, exist_ok=True)
    plot_instance.savefig(directory + "/" + name + ".svg")
    plot_instance.savefig(directory + "/" + name + ".pdf")


def save_evaluator(file, identifier):
    directory = "figs/eval_" + identifier
    Path(directory).mkdir(parents=True, exist_ok=True)
    shutil.copy(file, directory + "/" + ntpath.basename(file) + ".txt")
