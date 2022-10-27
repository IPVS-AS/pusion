import numpy as np
from sklearn import preprocessing, model_selection
from collections import Counter
from matplotlib import pyplot as plt

def preprocess_data_kfold_cross_validation(data: np.ndarray = None,
                                           labels: np.ndarray = None,
                                           n_splits=5,
                                           transformation_type: str = 'normalization',
                                           mapping_range: list = [-1, 1]) -> [np.ndarray, np.ndarray]:
    '''
        :param data: input data, i. e., the feature vector
        :param labels: label vector y
        :param n_splits: splits for folds of k fold cross validation
        :param transformation_type: can be 'normalization' or 'standardization' for specifying the transformation type
        :param mapping_range: the normalization range if :param transformation_type = 'normalization' is selected, or the z-Score for :param transformation_type
        'standardization'
        :return: a list of preprocessed input data and corresponding labels
        '''

    # preprocessing (normalization or standardization)
    preprocessed_data = []
    if transformation_type == "normalization":
        scaler = preprocessing.MinMaxScaler(mapping_range).fit(data)
        preprocessed_data = scaler.transform(data)
    elif transformation_type == "standardization":
        scaler = preprocessing.StandardScaler().fit(data)
        preprocessed_data = scaler.transform(data)

    # label vector transformation for sklearn
    number_labels = np.argmax(labels, axis=1)

    skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
    folds = skf.split(preprocessed_data, number_labels)

    train_indices_of_folds = []
    test_indices_of_folds = []
    X_folds = {}
    Y_folds = {}

    i_num = 0
    for train, test in folds:
        train_indices_of_folds.append(train)
        test_indices_of_folds.append(test)
        X_folds[i_num] = preprocessed_data[test]
        Y_folds[i_num] = labels[test]
        i_num+=1

    return X_folds, Y_folds



def preprocess_data(data: np.ndarray = None,
                    labels: np.ndarray = None,
                    splitting_ratios: list = [80, 10, 10],
                    transformation_type: str = 'normalization',
                    mapping_range: list = [-1, 1]) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray]:
    '''
    :param data: input data, i. e., the feature vector
    :param labels: label vector y
    :param splitting_ratios: ratios to split :param data into train, validation and test data sets
    :param transformation_type: can be 'normalization' or 'standardization' for specifying the transformation type
    :param mapping_range: the normalization range if :param transformation_type = 'normalization' is selected, or the z-Score for :param transformation_type
    'standardization'
    :return: a list of preprocessed input data and corresponding labels
    '''

    # preprocessing (normalization or standardization)
    preprocessed_data = []
    if transformation_type == "normalization":
        scaler = preprocessing.MinMaxScaler(mapping_range).fit(data)
        preprocessed_data = scaler.transform(data)
        pass
    elif transformation_type == "standardization":
        scaler = preprocessing.StandardScaler().fit(data)
        preprocessed_data = scaler.transform(data)
        pass

    # split dataset into training, validation and test set
    val_split = splitting_ratios[1] * 0.01
    test_split = splitting_ratios[2] * 0.01
    x_rest, x_val, y_rest, y_val = model_selection.train_test_split(preprocessed_data, labels,
                                                                    test_size=val_split, stratify=labels)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_rest, y_rest,
                                                                        test_size=test_split, stratify=y_rest)

    return x_train, x_val, x_test, y_train, y_val, y_test


def plot_class_distribution(labels: np.ndarray = None, class_label_list: np.ndarray = None, lm_type: str = 'lm1',
                            dataset_type: str = 'train'):
    '''
    :param labels: label vector y
    :param class_label_list: list of class names encoded in :param labels
    :param lm_type: type of label mapping used (lm1 to lm4)
    :param dataset_type: train/ validation/ test set
    :return:
    '''
    # make bar labels shorter
    list = []
    for row in labels:
        name = ""
        for entry in range(labels.shape[1]):
            if row[entry] == 1:
                name = name + "_" + class_label_list[entry]
        name = name.replace("_fault_lm1", "").replace("_class_lm1", "").replace("_fault_lm2", "") \
            .replace("_class_lm2", "").replace("_fault_lm3", "").replace("_class_lm3", "").replace("_fault_lm4", "") \
            .replace("_class_lm4", "").replace("_condition", "").replace("_accumulator", "").replace("_internal", "")
        if name[0] == "_":
            name = name[1:]
        list.append(name)

    # make plot
    list.sort()
    counts = Counter(list)
    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.title("Distribution of " + dataset_type + " for " + lm_type, fontsize=14)
    plt.tight_layout()
    plt.show()
