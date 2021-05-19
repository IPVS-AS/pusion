import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from pusion import *
from pusion.input_output.file_input_output import *

n_classes = 5
n_samples = 2000
random_state = 1

np.random.seed(random_state)

# ======================================================================================================================

classifiers = [
    MLPClassifier(max_iter=5000, random_state=1),
    MLPClassifier(max_iter=5000, random_state=2),
    MLPClassifier(max_iter=5000, random_state=3),
    MLPClassifier(max_iter=5000, random_state=4),
    MLPClassifier(max_iter=5000, random_state=5),
]

np.random.seed(random_state)
y_ensemble_valid, y_valid, y_ensemble_test, y_test = \
        generate_multiclass_ensemble_classification_outputs(classifiers, n_classes, n_samples, True)

y_ensemble_test = np.append(y_ensemble_test, y_ensemble_valid, axis=1)
y_test = np.append(y_test, y_valid, axis=0)

for i, classifier in enumerate(classifiers):
    data = {
        'Y_predictions': y_ensemble_test[i],
        'Y_test': y_test
    }
    dump_pusion_data(data, 'datasets/generated_multiclass_classification_classifier_' + str(i) + '.pickle')
