import numpy as np


def print_coverage_matrix(classification_coverage, n_classifier, n_classes):
    coverage_matrix = np.zeros((n_classifier, n_classes), dtype=int)
    for i in range(len(classification_coverage)):
        for j in classification_coverage[i]:
            coverage_matrix[i, j] = 1
    print(coverage_matrix)