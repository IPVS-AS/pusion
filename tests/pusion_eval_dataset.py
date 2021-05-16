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
    # 'datasets/Time-SE-ResNet_lr0.01_bs128_ep24_3.pickle',
    # 'datasets/Time-SE-ResNet_lr0.01_bs128_ep70_4.pickle',
    # 'datasets/Time-SE-ResNet_lr0.01_bs128_ep70_5.pickle',
]

data = load_native_files_as_data(dataset_files)

decision_outputs = [
    data[0]['Y_predictions'],
    data[1]['Y_predictions'],
    # data[2]['Y_predictions'],
    # data[3]['Y_predictions'],
    # data[4]['Y_predictions']
]

true_assignments = np.array(data[0]['Y_test'])

coverage = [
    [0,  1,  2,  3],
    [0,  1,  2,  3],
    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [0,  8]
]

cr = False

y_ensemble_valid, y_valid, y_ensemble_test, y_test = \
    split_into_train_and_validation_data(decision_outputs, true_assignments, validation_size=.75)

eval_metrics = [
    p.PerformanceMetric.ACCURACY,
    p.PerformanceMetric.F1_SCORE,
    p.PerformanceMetric.MEAN_CONFIDENCE,
]


print("============= Ensemble ===============")
eval_classifiers = Evaluation(*eval_metrics)
y_ensemble_test = multiclass_prediction_tensor_to_decision_tensor(y_ensemble_test)
if cr:
    eval_classifiers.set_instances(['Ensemble'])
    eval_classifiers.evaluate_cr_decision_outputs(y_test, y_ensemble_test, coverage)
else:
    eval_classifiers.set_instances([('Classifier ' + str(i)) for i in range(len(dataset_files))])
    eval_classifiers.evaluate(y_test, y_ensemble_test)

print(eval_classifiers.get_report())

# ---- GenericCombiner -------------------------------------------------------------------------------------------------
dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
if cr:
    dp.set_coverage(coverage)
dp.train(y_ensemble_valid, y_valid)
dp.combine(y_ensemble_test)

# ---- Evaluate all combiners
print("========== GenericCombiner ===========")
eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiners())
eval_combiner.set_runtimes(dp.get_multi_combiner_runtimes())
multi_comb_decision_outputs = multiclass_prediction_tensor_to_decision_tensor(dp.get_multi_combiner_decision_output())
if cr:
    eval_combiner.evaluate_cr_multi_combiner_decision_outputs(y_test, multi_comb_decision_outputs)
else:
    eval_combiner.evaluate(y_test, multi_comb_decision_outputs)
print(eval_combiner.get_report())

# ---- Report runtimes
print("------------- Runtimes ---------------")
print(eval_combiner.get_runtime_report())

print()
exit(0)
