from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from pusion import *
from pusion.input_output.file_input_output import *

n_classes = 5
n_samples = 2000

np.random.seed(1)

# ======================================================================================================================

ensembles = {
    0: {
        'ensemble_type': 'KNN',
        'classifiers': [
            KNeighborsClassifier(1),
            KNeighborsClassifier(3),
            KNeighborsClassifier(5),
            KNeighborsClassifier(7),
            KNeighborsClassifier(9),
        ]
    },
    1: {
        'ensemble_type': 'NN',
        'classifiers': [
            MLPClassifier(max_iter=5000, random_state=1),
            MLPClassifier(max_iter=5000, random_state=2),
            MLPClassifier(max_iter=5000, random_state=3),
            MLPClassifier(max_iter=5000, random_state=4),
            MLPClassifier(max_iter=5000, random_state=5),
        ]
    },
    2: {
        'ensemble_type': 'RF',
        'classifiers': [
            RandomForestClassifier(max_depth=1, n_estimators=10, random_state=1),
            RandomForestClassifier(max_depth=3, n_estimators=9, random_state=1),
            RandomForestClassifier(max_depth=5, n_estimators=8, random_state=1),
            RandomForestClassifier(max_depth=7, n_estimators=7, random_state=1),
            RandomForestClassifier(max_depth=10, n_estimators=6, random_state=1),
        ]
    }
}

# Multiclass
for i in ensembles:
    ensemble = ensembles[i]
    classifiers = ensemble['classifiers']
    np.random.seed(1)
    y_ensemble_valid, y_valid, y_ensemble_test, y_test = generate_multiclass_ensemble_classification_outputs(
        classifiers, n_classes, n_samples)
    ensemble['y_ensemble_valid'] = y_ensemble_valid
    ensemble['y_valid'] = y_valid
    ensemble['y_ensemble_test'] = y_ensemble_test
    ensemble['y_test'] = y_test

dump_pusion_data(ensembles, 'datasets/ensembles_generated_multiclass_classification.pickle')


# Multilabel
# for i in ensembles:
#     ensemble = ensembles[i]
#     classifiers = ensemble['classifiers']
#     y_ensemble_valid, y_valid, y_ensemble_test, y_test = generate_multilabel_ensemble_classification_outputs(
#         classifiers, n_classes, n_samples)
#     ensemble['y_ensemble_valid'] = y_ensemble_valid
#     ensemble['y_valid'] = y_valid
#     ensemble['y_ensemble_test'] = y_ensemble_test
#     ensemble['y_test'] = y_test

