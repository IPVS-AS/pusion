import time

import numpy as np

import pusion as p
from pusion.evaluation.evaluation import Evaluation
from pusion.input_output.file_input_output import load_native_files_as_data
from pusion.util.generator import split_into_train_and_validation_data
from pusion.util.transformer import *

dataset_files = [
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep24_1.pickle',
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep04_2.pickle',
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep24_3.pickle',
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep70_4.pickle',
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep70_5.pickle',
]

data = load_native_files_as_data(dataset_files)

decision_outputs = np.array([data[0]['Y_predictions'], data[1]['Y_predictions']])
true_assignments = np.array(data[0]['Y_test'])

y_ensemble_valid, y_valid, y_ensemble_test, y_test = \
    split_into_train_and_validation_data(decision_outputs, true_assignments, validation_size=.75)

eval_metrics = [
    p.PerformanceMetric.MEAN_CONFIDENCE,
    # p.PerformanceMetric.ACCURACY,
    # p.PerformanceMetric.F1_SCORE,
    # p.PerformanceMetric.PRECISION,
    # p.PerformanceMetric.RECALL
]


print("============= Ensemble ===============")
eval_classifiers = Evaluation(*eval_metrics)
eval_classifiers.set_instances(['ResNet_ep24', 'ResNet_ep04'])
# eval_classifiers.evaluate(y_test, multiclass_prediction_tensor_to_decision_tensor(y_ensemble_test))
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
# eval_combiner.evaluate(y_test, multiclass_prediction_tensor_to_decision_tensor(dp.get_multi_combiner_decision_output()))
eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())
print(eval_combiner.get_report())

# ---- Report runtimes
print("------------- Runtimes ---------------")
print(eval_combiner.get_runtime_report())
print("Total train time:", t_elapsed_train)
print("Total combine time:", t_elapsed_combine)

print()
exit(0)
