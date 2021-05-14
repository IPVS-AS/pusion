import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pusion as p
from pusion.core.combiner import UtilityBasedCombiner, TrainableCombiner
from pusion.evaluation.evaluation import Evaluation
from pusion.evaluation.evaluation_metrics import *
from pusion.input_output.file_input_output import *

warnings.filterwarnings('error')  # halt on warning

eval_id = time.strftime("%Y%m%d-%H%M%S")

perf_metrics = (p.PerformanceMetric.ACCURACY, p.PerformanceMetric.F1_SCORE, p.PerformanceMetric.MEAN_CONFIDENCE)


# data1 = load_native_files_as_data(['datasets/ensembles_generated_multiclass_classification.pickle'])
# data2 = load_native_files_as_data(['datasets/ensembles_generated_multilabel_classification.pickle'])
# data3 = load_native_files_as_data(['datasets/ensembles_generated_cr_multiclass_classification.pickle'])
# data4 = load_native_files_as_data(['datasets/ensembles_generated_cr_multiclass_classification.pickle'])

data = load_native_files_as_data(['datasets/ensembles_generated_multiclass_classification.pickle'])[0]

# Ensemble data
ensembles = data['ensembles']
n_classes = data['n_classes']
n_samples = data['n_samples']
random_state = data['random_state']

# ----------------------------------------------------------------------------------------------------------------------

ensemble_wise_types = []
ensemble_wise_accuracies = []
ensemble_wise_max_accuracy = []
ensemble_wise_mean_accuracy = []


np.random.seed(random_state)

for i in ensembles:
    ensemble = ensembles[i]

    y_ensemble_valid = ensemble['y_ensemble_valid']
    y_valid = ensemble['y_valid']
    y_ensemble_test = ensemble['y_ensemble_test']
    y_test = ensemble['y_test']

    print("============== Ensemble ================")
    eval_ensemble = Evaluation(*perf_metrics)
    eval_ensemble.set_instances(ensemble['classifiers'])
    eval_ensemble.evaluate(y_test, y_ensemble_test)
    print(eval_ensemble.get_report())

    print("=========== GenericCombiner ============")
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
    dp.set_parallel(True)

    dp.train(y_ensemble_valid, y_valid)
    y_comb = dp.combine(y_ensemble_test)

    eval_combiner = Evaluation()
    eval_combiner.set_metrics(*perf_metrics)
    eval_combiner.set_instances(dp.get_combiners())
    eval_combiner.evaluate(y_test, y_comb)
    print(eval_combiner.get_report())
    print("----------------------------------------")
    eval_combiner.set_runtimes(dp.get_multi_combiner_runtimes())
    print(eval_combiner.get_runtime_report())
    print("----------------------------------------")
    print()

    # ------------------------------------------------------------------------------------------------------------------
    ensemble_wise_types.append(ensemble['ensemble_type'])
    ensemble_wise_max_accuracy.append(eval_ensemble.get_top_n_instances(n=1)[0][1])

    ensemble_accuracies = [p[1] for p in eval_ensemble.get_top_n_instances(metric=p.PerformanceMetric.ACCURACY)]
    ensemble_wise_accuracies.append(ensemble_accuracies)

    ensemble_mean_accuracy = np.mean(ensemble_accuracies)
    ensemble_wise_mean_accuracy.append(ensemble_mean_accuracy)


# === Plots ============================================================================================================
meanprops = dict(markerfacecolor='black', markeredgecolor='white')

# --- Ensemble max. accuracy -------------------------------------------------------------------------------------------
plt.figure()
plt.bar(ensemble_wise_types, ensemble_wise_max_accuracy, color='#006aba')
plt.ylabel("Max. Trefferquote", labelpad=15)
plt.tight_layout()
save(plt, "000_ensemble_max_accuracy", eval_id)
plt.close()

plt.figure()
plt.boxplot(ensemble_wise_accuracies, showmeans=True, meanprops=meanprops)
plt.ylabel("Trefferquote", labelpad=15)
plt.xticks(np.arange(1, len(ensemble_wise_types)+1), ensemble_wise_types)
plt.tight_layout()
save(plt, "001_ensemble_accuracies", eval_id)
plt.close()


# ======================================================================================================================
save_evaluator(__file__, eval_id)
print("Evaluation", eval_id, "done.")
