import time
import warnings

from sklearn.neural_network import MLPClassifier

import pusion as p
from pusion.evaluation.evaluation import Evaluation

warnings.filterwarnings('error')  # halt on error


# ---- Classifiers -----------------------------------------------------------------------------------------------------
classifiers = [
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(10,)),  # MLK
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10)),  # MLK
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10, 10)),  # MLK
]

y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multilabel_ensemble_classification_outputs(
    classifiers, n_classes=5, n_samples=10000)


eval_metrics = [
    p.PerformanceMetric.ACCURACY,
    p.PerformanceMetric.MICRO_F1_SCORE,
    p.PerformanceMetric.MICRO_PRECISION,
    p.PerformanceMetric.MICRO_RECALL
]


print("============= Ensemble ===============")
eval_classifiers = Evaluation(*eval_metrics)
eval_classifiers.set_instances(classifiers)
eval_classifiers.evaluate(y_test, y_ensemble_test)
print(eval_classifiers.get_report())

# ---- GenericCombiner -------------------------------------------------------------------------------------------------
dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))

t_begin = time.perf_counter()
dp.train(y_ensemble_valid, y_valid)
t_elapsed_train = time.perf_counter() - t_begin

t_begin = time.perf_counter()
dp.combine(y_ensemble_test)
t_elapsed_combine = time.perf_counter() - t_begin

# ---- Evaluate all combiners
print("========== GenericCombiner ===========")
eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiners())
eval_combiner.set_runtimes(dp.get_multi_combiner_runtimes())
eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())
print(eval_combiner.get_report())

# ---- Report runtimes
print("------------- Runtimes ---------------")
print(eval_combiner.get_runtime_report())
print("Total train time:", t_elapsed_train)
print("Total combine time:", t_elapsed_combine)


# ---- AutoCombiner ----------------------------------------------------------------------------------------------------
# dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))
#
# t_begin = time.perf_counter()
# dp.train(y_ensemble_valid, y_valid)
# t_elapsed_train = time.perf_counter() - t_begin
#
# t_begin = time.perf_counter()
# y_comb = dp.combine(y_ensemble_test)
# t_elapsed_combine = time.perf_counter() - t_begin
#
# # ---- Evaluate AutoCombiner
# print("============ AutoCombiner ============")
# eval_combiner = Evaluation(*eval_metrics)
# eval_combiner.set_instances(dp.get_combiner())
# eval_combiner.evaluate(y_test, y_comb)
# print(eval_combiner.get_report())
#
# # ---- Evaluate internal combiners
# print("--------------------------------------")
# eval_combiner = Evaluation(*eval_metrics)
# eval_combiner.set_instances(dp.get_combiners())
# eval_combiner.set_runtimes(dp.get_multi_combiner_runtimes())
# eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())
# print(eval_combiner.get_report())
#
# # ---- Report runtimes
# print("------------- Runtimes ---------------")
# print(eval_combiner.get_runtime_report())
# print("Total train time:", t_elapsed_train)
# print("Total combine time:", t_elapsed_combine)
# print()
