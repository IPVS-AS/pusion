import warnings

from sklearn.neural_network import MLPClassifier

import pusion as p
from pusion.evaluation.evaluation import Evaluation

warnings.filterwarnings('error')  # halt on error


classifiers = [
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(100,)),
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(100, 50)),
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(100, 50, 25)),
]

eval_metrics = [
    p.PerformanceMetric.ACCURACY,
    p.PerformanceMetric.F1_SCORE,
    p.PerformanceMetric.PRECISION,
    p.PerformanceMetric.RECALL
]

coverage = p.generate_classification_coverage(n_classifiers=len(classifiers), n_classes=5, overlap=.6, normal_class=True)

y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multilabel_cr_ensemble_classification_outputs(
    classifiers=classifiers, n_classes=5, n_samples=2000, coverage=coverage)


print("============= Ensemble ===============")
eval_classifiers = Evaluation(*eval_metrics)
eval_classifiers.set_instances("Ensemble")
eval_classifiers.evaluate_cr_decision_outputs(y_test, y_ensemble_test, coverage)
print(eval_classifiers.get_report())

print("============ AutoCombiner ============")
dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))
dp.set_parallel(False)
dp.set_coverage(coverage)
dp.train(y_ensemble_valid, y_valid)
y_comb = dp.combine(y_ensemble_test)

eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiner())
eval_combiner.evaluate_cr_decision_outputs(y_test, [y_comb])
print(eval_combiner.get_report())
print("Selected:", type(dp.get_optimal_combiner()).__name__)

print("--------------------------------------")
eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiners())
eval_combiner.evaluate_cr_multi_combiner_decision_outputs(y_test, dp.get_multi_combiner_decision_output())
print(eval_combiner.get_report())

print("========== GenericCombiner ===========")
dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
dp.set_coverage(coverage)
dp.train(y_ensemble_valid, y_valid)
dp.combine(y_ensemble_test)

eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiners())
eval_combiner.evaluate_cr_multi_combiner_decision_outputs(y_test, dp.get_multi_combiner_decision_output())
print(eval_combiner.get_report())
