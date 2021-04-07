import uuid
import warnings
import pusion as p

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pusion.control.evaluation import Evaluation
from pusion.evaluation.evaluation_metrics import *
from pusion.input_output.file_input_output import *

warnings.filterwarnings('error')  # TODO delete

eval_id = uuid.uuid4().hex
n_runs = 1


classifiers_performance_run_tuples = []
combiners_performance_run_tuples = []
performance_improvements = []
classifier_max_scores = []
combiners_max_scores = []
classifier_score_stds = []
ensemble_diversity_cohens_kappa_scores = []
ensemble_diversity_correlation_scores = []
ensemble_diversity_q_statistic_scores = []


for i in range(n_runs):
    classifiers = [
        # KNeighborsClassifier(3),
        # KNeighborsClassifier(5),
        # KNeighborsClassifier(10),
        # KNeighborsClassifier(20),
        # KNeighborsClassifier(30),
        # DecisionTreeClassifier(max_depth=5),  # MLK
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),  # MLK
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(10,)),  # MLK
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10)),  # MLK
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10, 10)),  # MLK
        # LinearDiscriminantAnalysis(),
        # LogisticRegression(),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
    ]

    coverage = p.generate_classification_coverage(n_classifier=len(classifiers),
                                                  n_classes=4,
                                                  overlap=1,
                                                  normal_class=True)

    y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multilabel_ensemble_classification_outputs(
        classifiers, n_classes=4, n_samples=10000, coverage=None)

    print("============== Ensemble ================")
    eval_classifiers = Evaluation()
    eval_classifiers.set_metrics(p.PerformanceMetric.ACCURACY,
                                 p.PerformanceMetric.F1_SCORE,
                                 p.PerformanceMetric.PRECISION,
                                 p.PerformanceMetric.RECALL)
    eval_classifiers.set_instances(classifiers)

    eval_classifiers.evaluate(y_test, y_ensemble_test)

    print("=========== GenericCombiner ============")
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))

    dp.train(y_ensemble_valid, y_valid)
    y_comb = dp.combine(y_ensemble_test)

    eval_combiner = Evaluation()
    eval_combiner.set_metrics(p.PerformanceMetric.ACCURACY)
    eval_combiner.set_instances(dp.get_combiners())
    eval_combiner.evaluate(y_test, y_comb)

    print("========================================")

    classifiers_performance_tuples = eval_classifiers.get_top_n_instances()
    classifiers_performance_run_tuples.append(classifiers_performance_tuples)

    combiners_performance_tuples = eval_combiner.get_top_n_instances()
    combiners_performance_run_tuples.append(combiners_performance_tuples)

    classifier_max_score = classifiers_performance_tuples[0][1]
    classifier_max_scores.append(classifier_max_score)

    combiners_max_score = combiners_performance_tuples[0][1]
    combiners_max_scores.append(combiners_max_score)

    performance_improvement = combiners_max_score - classifier_max_score
    performance_improvements.append(performance_improvement)

    classifier_score_std = np.std([t[1] for t in classifiers_performance_tuples])
    classifier_score_stds.append(classifier_score_std)

    ensemble_diversity_cohens_kappa_scores.append(pairwise_cohens_kappa_multiclass(y_ensemble_test))
    ensemble_diversity_correlation_scores.append(correlation(y_ensemble_test))
    ensemble_diversity_q_statistic_scores.append(q_statistic(y_ensemble_test))


# Diversity -- Framework Performance
plt.plot(ensemble_diversity_cohens_kappa_scores, combiners_max_scores, 'ro')
plt.xlabel("Diversity (Cohen's Kappa)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "data_plot_a_div_cohens_kappa__framework_performance", eval_id)
plt.close()

plt.plot(ensemble_diversity_correlation_scores, combiners_max_scores, 'bs')
plt.xlabel("Diversity (Correlation)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "data_plot_a_div_correlation__framework_performance", eval_id)
plt.close()

plt.plot(ensemble_diversity_q_statistic_scores, combiners_max_scores, 'g^')
plt.xlabel("Diversity (Q-statistic)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "data_plot_a_div_q_stat__framework_performance", eval_id)
plt.close()


# Diversity -- Performance Improvement
plt.plot(ensemble_diversity_cohens_kappa_scores, performance_improvements, 'ro')
plt.xlabel("Diversity (Cohen's Kappa)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "data_plot_b_div_cohens_kappa__p_imp", eval_id)
plt.close()

plt.plot(ensemble_diversity_correlation_scores, performance_improvements, 'bs')
plt.xlabel("Diversity (Correlation)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "data_plot_b_div_correlation__p_imp", eval_id)
plt.close()

plt.plot(ensemble_diversity_q_statistic_scores, performance_improvements, 'g^')
plt.xlabel("Diversity (Q-statistic)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "data_plot_b_div_q_stat__p_imp", eval_id)
plt.close()


# Performance comparison (Ensemble/Framework)
plt.boxplot([classifier_max_scores, combiners_max_scores])
plt.title("Performance comparison (" + str(n_runs) + " runs)")
plt.ylabel("Max. Accuracy", labelpad=15)
plt.xticks([1, 2], ['Ensemble', 'Framework'])
plt.tight_layout()
save(plt, "box_plot_max_performance_comparison", eval_id)
plt.close()

# Performance improvement by Framework
plt.boxplot(performance_improvements)
plt.title("Performance improvement (" + str(n_runs) + " runs)")
plt.ylabel("Accuracy (difference)", labelpad=15)
plt.xticks([1], ['Framework'])
plt.tight_layout()
save(plt, "box_plot_performance_improvement", eval_id)
plt.close()

# Fusion methods comparison
reduced_combiners_performances = {}
for tuples in combiners_performance_run_tuples:  # reduce
    for t in tuples:
        comb = type(t[0])
        if comb not in reduced_combiners_performances:
            reduced_combiners_performances[comb] = []
        reduced_combiners_performances[comb].append(t[1])

combiners = [comb for comb in reduced_combiners_performances.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_performances = [reduced_combiners_performances[c] for c in combiners]

plt.figure(figsize=(10, 4.8))
plt.boxplot(combiners_performances)
plt.title("Fusion methods comparison (" + str(n_runs) + " runs)")
plt.ylabel("Accuracy", labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
plt.tight_layout()
save(plt, "box_plot_combiner_comparison", eval_id)
plt.close()

# Fusion methods comparison (with control)
combiners_performances.append(classifier_max_scores)
combiners_names.append('Control')

plt.figure(figsize=(10, 4.8))
plt.boxplot(combiners_performances)
plt.title("Framework control comparison (" + str(n_runs) + " runs)")
plt.ylabel("Accuracy", labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
plt.tight_layout()
save(plt, "box_plot_combiner_control_comparison", eval_id)
plt.close()


# Ensemble Standard Deviation (Scatter-Plot)
fig, ax = plt.subplots()
scatter = ax.scatter(classifier_score_stds, combiners_max_scores, c=ensemble_diversity_q_statistic_scores)
ax.set_xlabel('Ensemble Standard Deviation (Accuracy)', labelpad=15)
ax.set_ylabel('Framework Performance (Accuracy)', labelpad=15)
fig.colorbar(scatter).set_label("Diversity (Q-statistic)", labelpad=15)
plt.tight_layout()
save(plt, "scatter_plot_cls_stds__framework_performance__diversity_q_stat", eval_id)
plt.close()


print("Evaluation", eval_id, "done.")
