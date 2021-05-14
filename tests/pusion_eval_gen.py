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

multi_ensemble_data = {
    'n_classes': n_classes,
    'n_samples': n_samples,
    'random_state': random_state,
    'ensembles': {
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
}


# --- Multiclass -- redundant ------------------------------------------------------------------------------------------
for i in multi_ensemble_data['ensembles']:
    ensemble = multi_ensemble_data['ensembles'][i]
    classifiers = ensemble['classifiers']
    np.random.seed(random_state)
    ensemble['y_ensemble_valid'], ensemble['y_valid'], ensemble['y_ensemble_test'], ensemble['y_test'] = \
        generate_multiclass_ensemble_classification_outputs(classifiers, n_classes, n_samples)

dump_pusion_data(multi_ensemble_data, 'datasets/ensembles_generated_multiclass_classification.pickle')


# --- Multilabel -- redundant ------------------------------------------------------------------------------------------
for i in multi_ensemble_data['ensembles']:
    ensemble = multi_ensemble_data['ensembles'][i]
    classifiers = ensemble['classifiers']
    np.random.seed(random_state)
    ensemble['y_ensemble_valid'], ensemble['y_valid'], ensemble['y_ensemble_test'], ensemble['y_test'] = \
        generate_multilabel_ensemble_classification_outputs(classifiers, n_classes, n_samples)

dump_pusion_data(multi_ensemble_data, 'datasets/ensembles_generated_multilabel_classification.pickle')


# --- Multiclass -- complementary-redundant ----------------------------------------------------------------------------
for i in multi_ensemble_data['ensembles']:
    ensemble = multi_ensemble_data['ensembles'][i]
    classifiers = ensemble['classifiers']
    np.random.seed(random_state)
    coverage = generate_classification_coverage(len(classifiers), n_classes, .7, True)
    np.random.seed(random_state)
    ensemble['y_ensemble_valid'], ensemble['y_valid'], ensemble['y_ensemble_test'], ensemble['y_test'] = \
        generate_multiclass_cr_ensemble_classification_outputs(classifiers, n_classes, n_samples, coverage)
    ensemble['coverage'] = coverage

dump_pusion_data(multi_ensemble_data, 'datasets/ensembles_generated_cr_multiclass_classification.pickle')


# --- Multilabel -- complementary-redundant ----------------------------------------------------------------------------
for i in multi_ensemble_data['ensembles']:
    ensemble = multi_ensemble_data['ensembles'][i]
    classifiers = ensemble['classifiers']
    np.random.seed(random_state)
    coverage = generate_classification_coverage(len(classifiers), n_classes, .7, True)
    np.random.seed(random_state)
    ensemble['y_ensemble_valid'], ensemble['y_valid'], ensemble['y_ensemble_test'], ensemble['y_test'] = \
        generate_multilabel_cr_ensemble_classification_outputs(classifiers, n_classes, n_samples, coverage)
    ensemble['coverage'] = coverage

dump_pusion_data(multi_ensemble_data, 'datasets/ensembles_generated_cr_multilabel_classification.pickle')
