import pickle


def load_native_files_as_data(file_paths):
    data = []
    for file_path in file_paths:
        with (open(file_path, "rb")) as handle:
            data.append(pickle.load(handle))
    return data


def dump_clunpy_fusion_data(data, file_path='fusion_output.pickle'):
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
