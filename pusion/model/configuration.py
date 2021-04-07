from pusion.util.constants import *


class Configuration:
    def __init__(self, method,
                 problem=Problem.GENERIC,
                 assignment_type=AssignmentType.GENERIC,
                 coverage_type=CoverageType.GENERIC):

        self.method = method
        self.problem = problem
        self.assignment_type = assignment_type
        self.coverage_type = coverage_type

    def get_tuple(self):
        return self.method, self.problem, self.assignment_type, self.coverage_type

    def get_pac(self):
        return self.problem, self.assignment_type, self.coverage_type
