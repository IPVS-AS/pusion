from pusion.util.constants import *
from pusion.util.transformer import *


def determine_problem(decision_outputs):
    if isinstance(decision_outputs, np.ndarray):
        return __determine_tensor_problem(decision_outputs)

    problem_list = []
    for dt in decision_outputs:
        problem_list.append(__determine_tensor_problem(np.array(dt)))
    if np.all(np.char.strip(problem_list) == Problem.MULTI_CLASS):
        return Problem.MULTI_CLASS
    return Problem.MULTI_LABEL


def __determine_tensor_problem(decision_tensor):
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
    if isinstance(decision_outputs, np.ndarray):
        return __determine_tensor_assignment_type(decision_outputs)

    assignment_type_list = []
    for dt in decision_outputs:
        assignment_type_list.append(__determine_tensor_assignment_type(np.array(dt)))
    if np.all(np.char.strip(assignment_type_list) == AssignmentType.CRISP):
        return AssignmentType.CRISP
    return AssignmentType.CONTINUOUS


def __determine_tensor_assignment_type(decision_tensor):
    if np.all(decision_tensor <= 0) or np.any(decision_tensor > 1):
        raise TypeError("The assignment type could not be determined for the given decision tensor.")

    if np.all(np.logical_or(decision_tensor == 0, decision_tensor == 1)):
        return AssignmentType.CRISP

    if np.all(np.logical_or(decision_tensor >= 0, decision_tensor <= 1)):
        return AssignmentType.CONTINUOUS


def determine_coverage_type(coverage):
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
    if not coverage:
        if not isinstance(decision_outputs, np.ndarray):
            raise TypeError('A non-numpy array indicating a possible complementary or complementary-redundant coverage '
                            'is given without coverage specification.')
        return determine_problem(decision_outputs), determine_assignment_type(decision_outputs), CoverageType.REDUNDANT
    else:
        return determine_problem(decision_outputs), \
               determine_assignment_type(decision_outputs), \
               determine_coverage_type(coverage)


def split_into_train_and_validation_data(decision_tensor, true_assignment, validation_size=0.5):
    n_validation_samples = int(len(true_assignment) * validation_size)
    all_indices = np.arange(len(true_assignment))
    validation_indices = np.random.choice(all_indices, n_validation_samples, replace=False)
    mask = np.ones(len(all_indices), bool)
    mask[validation_indices] = 0
    train_indices = all_indices[mask]
    true_assignment_train = true_assignment[train_indices]
    true_assignment_validation = true_assignment[validation_indices]
    decision_tensor_train = []
    decision_tensor_validation = []

    for decision_matrix in decision_tensor:
        decision_tensor_train.append(decision_matrix[train_indices])
        decision_tensor_validation.append(decision_matrix[validation_indices])

    return decision_outputs_to_decision_tensor(decision_tensor_train), \
        decision_outputs_to_decision_tensor(true_assignment_train), \
        decision_outputs_to_decision_tensor(decision_tensor_validation), \
        decision_outputs_to_decision_tensor(true_assignment_validation)
