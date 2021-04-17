from pusion.util.constants import *
from pusion.util.transformer import *


def determine_problem(decision_outputs):
    """
    Determine the classification problem based on the decision outputs.

    :param decision_outputs: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
            `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
            due to the coverage.
    :return: `string` constant `'MULTI_CLASS'` or `'MULTI_LABEL'`. See `pusion.util.constants.Problem`.
    """
    if isinstance(decision_outputs, np.ndarray):
        return __determine_tensor_problem(decision_outputs)

    problem_list = []
    for dt in decision_outputs:
        problem_list.append(__determine_tensor_problem(np.array(dt)))
    if np.all(np.char.strip(problem_list) == Problem.MULTI_CLASS):
        return Problem.MULTI_CLASS
    return Problem.MULTI_LABEL


def __determine_tensor_problem(decision_tensor):
    """
    Helper method for ``determine_problem`` to determine the classification problem on `numpy.array` tensors.
    """
    # sum up all decisions along classes
    decision_sum = np.sum(decision_tensor, axis=decision_tensor.ndim - 1)
    assignment_type = determine_assignment_type(decision_tensor)
    if assignment_type == AssignmentType.CRISP:
        if np.all(decision_sum == 1):
            return Problem.MULTI_CLASS
        if np.all(decision_sum >= 1):
            return Problem.MULTI_LABEL

    if assignment_type == AssignmentType.CONTINUOUS:
        if np.all(decision_sum == 1):
            return Problem.MULTI_CLASS
        else:
            return Problem.MULTI_LABEL
    raise TypeError("The problem could not be determined for the given decision tensor.")


def determine_assignment_type(decision_outputs):
    """
    Determine the assignment type based on the decision outputs.

    :param decision_outputs: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
            `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
            due to the coverage.
    :return: `string` constant `'CRISP'` or `'CONTINUOUS'`. See `pusion.util.constants.AssignmentType`.
    """
    if isinstance(decision_outputs, np.ndarray):
        return __determine_tensor_assignment_type(decision_outputs)

    assignment_type_list = []
    for dt in decision_outputs:
        assignment_type_list.append(__determine_tensor_assignment_type(np.array(dt)))
    if np.all(np.char.strip(assignment_type_list) == AssignmentType.CRISP):
        return AssignmentType.CRISP
    return AssignmentType.CONTINUOUS


def __determine_tensor_assignment_type(decision_tensor):
    """
    Helper method for ``determine_assignment_type`` to determine the assignment type on `numpy.array` tensors.
    """
    if np.all(decision_tensor <= 0) or np.any(decision_tensor > 1):
        raise TypeError("The assignment type could not be determined for the given decision tensor.")

    if np.all(np.logical_or(decision_tensor == 0, decision_tensor == 1)):
        return AssignmentType.CRISP

    if np.all(np.logical_or(decision_tensor >= 0, decision_tensor <= 1)):
        return AssignmentType.CONTINUOUS


def determine_coverage_type(coverage):
    """
    Determine the coverage type.

    :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a classifier,
            which is identified by the positional index of the respective list.
    :return: `string` constant `'REDUNDANT'`, `'COMPLEMENTARY'`  or `'COMPLEMENTARY_REDUNDANT'`.
            See `pusion.util.constants.CoverageType`.
    """

    # determine coverage sets
    omega = [set(classes) for classes in coverage]
    # check for redundancy
    equality = [omega[i] == omega[j] for i in range(len(omega)) for j in range(i)]
    if np.all(equality):
        return CoverageType.REDUNDANT
    # check for complementary
    disjunction = [omega[i].intersection(omega[j]) == set() for i in range(len(omega)) for j in range(i)]
    if np.all(disjunction):
        return CoverageType.COMPLEMENTARY
    # complementary-redundant, since neither redundancy nor complementary rules apply
    return CoverageType.COMPLEMENTARY_REDUNDANT


def determine_pac(decision_outputs, coverage=None):
    """
    Determine the PAC-tuple (problem, assignment type and coverage type) based on the given decision outputs and
    coverage.

    :param decision_outputs: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
            `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
            due to the coverage.
    :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a classifier,
            which is identified by the positional index of the respective list.
    :return: `tuple` of string constants representing the PAC. See `pusion.util.constants.Problem`,
            `pusion.util.constants.AssignmentType` and `pusion.util.constants.CoverageType`.
    """
    if not coverage:
        if not isinstance(decision_outputs, np.ndarray):
            raise TypeError('A non-numpy array indicating a possible complementary or complementary-redundant coverage '
                            'is given without coverage specification.')
        return determine_problem(decision_outputs), determine_assignment_type(decision_outputs), CoverageType.REDUNDANT
    else:
        return determine_problem(decision_outputs), \
               determine_assignment_type(decision_outputs), \
               determine_coverage_type(coverage)


def split_into_train_and_validation_data(decision_tensor, true_assignments, validation_size=0.5):
    """
    Split the decision outputs (tensor) from multiple classifiers as well as the true assignments randomly into train
    and validation datasets.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of true class assignments.
    :param validation_size: Proportion between `0` and `1` for the size of the validation data set.
    :return: `tuple` of
            (1) `numpy.array` of shape `(n_classifiers, n_samples', n_classes)`,
            (2) `numpy.array` of shape `(n_classifiers, n_samples')`,
            (3) `numpy.array` of shape `(n_classifiers, n_samples'', n_classes)`,
            (4) `numpy.array` of shape `(n_classifiers, n_samples'')`, with `n_samples'` as the number of training
            samples and `n_samples''` as the number of validation samples.
    """
    n_validation_samples = int(len(true_assignments) * validation_size)
    all_indices = np.arange(len(true_assignments))
    validation_indices = np.random.choice(all_indices, n_validation_samples, replace=False)
    mask = np.ones(len(all_indices), bool)
    mask[validation_indices] = 0
    train_indices = all_indices[mask]
    true_assignments_train = true_assignments[train_indices]
    true_assignments_validation = true_assignments[validation_indices]
    decision_tensor_train = []
    decision_tensor_validation = []

    for decision_matrix in decision_tensor:
        decision_tensor_train.append(decision_matrix[train_indices])
        decision_tensor_validation.append(decision_matrix[validation_indices])

    return decision_outputs_to_decision_tensor(decision_tensor_train), \
        decision_outputs_to_decision_tensor(true_assignments_train), \
        decision_outputs_to_decision_tensor(decision_tensor_validation), \
        decision_outputs_to_decision_tensor(true_assignments_validation)
