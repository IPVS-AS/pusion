from sklearn.neural_network import MLPClassifier

import pusion as p
from pusion.evaluation.evaluation import Evaluation

classifiers = [
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(10,)),  # MLK
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10)),  # MLK
    MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10, 10)),  # MLK
]

eval_metrics = [
    p.PerformanceMetric.ACCURACY,
    p.PerformanceMetric.F1_SCORE,
    p.PerformanceMetric.PRECISION,
    p.PerformanceMetric.RECALL
]

y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multilabel_ensemble_classification_outputs(
    classifiers, n_classes=4, n_samples=1000)

print("============= Ensemble ===============")
eval_classifiers = Evaluation(*eval_metrics)
eval_classifiers.set_instances(classifiers)
eval_classifiers.evaluate(y_test, y_ensemble_test)
print(eval_classifiers.get_report())

print("============ AutoCombiner ============")
dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))
dp.train(y_ensemble_valid, y_valid)
y_comb = dp.combine(y_ensemble_test)

eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiner())
eval_combiner.evaluate(y_test, y_comb)
print(eval_combiner.get_report())
print("--------------------------------------")

eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiners())
eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())
print(eval_combiner.get_report())

print("========== GenericCombiner ===========")
dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
dp.train(y_ensemble_valid, y_valid)
dp.combine(y_ensemble_test)

eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiners())
eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())
print(eval_combiner.get_report())
