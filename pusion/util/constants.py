class Problem:
    MULTI_LABEL = 'MULTI_LABEL'
    MULTI_CLASS = 'MULTI_CLASS'
    GENERIC = 'GENERIC'


class AssignmentType:
    CRISP = 'CRISP'
    CONTINUOUS = 'CONTINUOUS'
    GENERIC = 'GENERIC'


class CoverageType:
    REDUNDANT = 'REDUNDANT'
    COMPLEMENTARY = 'COMPLEMENTARY'
    COMPLEMENTARY_REDUNDANT = 'COMPLEMENTARY_REDUNDANT'
    GENERIC = 'GENERIC'


class EvidenceType:
    CONFUSION_MATRIX = 'CONFUSION_MATRIX'
    ACCURACY = 'ACCURACY'
    GENERIC = 'GENERIC'


class PAC:
    GENERIC = (Problem.GENERIC, AssignmentType.GENERIC, CoverageType.GENERIC)
