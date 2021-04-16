from pusion.util.constants import *


class Configuration:
    """
    The :class:`Configuration` forms the main parameter of the decision fusion framework. Based on this, the framework
    is able to check the compatibility of a respective decision fusion method to the given decision outputs.
    A configuration may be defined by the user or auto-detected by the framework.

    :param method: An explicit method provided by the framework which should be applied on the input data set.
            See `pusion.Method` for possible options.
    :param problem: Input problem type. See `pusion.util.constants.Problem` for possible options.
    :param assignment_type: The class assignment describes memberships to each individual class for a sample.
            A class assignment type is either crisp or continuous. Crisp assignments are equivalent to labels and
            continuous assignments represent probabilities for each class being true.
            See `pusion.util.constants.AssignmentType` for possible options.
    :param coverage_type: The classification coverage states for each input classifier, which classes it is able to
            decide. A classifier ensemble may yield a redundant, complementary or complementary-redundant coverage.
            See `pusion.util.constants.CoverageType` for possible options.
    """
    def __init__(self, method,
                 problem=Problem.GENERIC,
                 assignment_type=AssignmentType.GENERIC,
                 coverage_type=CoverageType.GENERIC):

        self.method = method
        self.problem = problem
        self.assignment_type = assignment_type
        self.coverage_type = coverage_type

    def get_tuple(self):
        """
        :return: A `tuple` of method, problem, assignment type and coverage type.
        """
        return self.method, self.problem, self.assignment_type, self.coverage_type

    def get_pac(self):
        """
        :return: A `tuple` of problem, assignment type and coverage type. This tuple is also referred to as PAC.
        """
        return self.problem, self.assignment_type, self.coverage_type
