class Problem:
    MULTI_LABEL = "multi_label"
    MULTI_CLASS = "multi_class"


class Format:
    NATIVE = "native"


class Method:
    BEHAVIOUR_KNOWLEDGE_SPACE = "behaviour_knowledge_space"
    BORDA_COUNT = "borda_count"
    COS_SIMILARITY = "cos_similarity"
    DECISION_TEMPLATES = "decision_templates"
    DEMPSTER_SHAFER = "dempster_shafer"
    MACRO_MAJORITY_VOTE = "macro_majority_vote"
    MICRO_MAJORITY_VOTE = "micro_majority_vote"
    NAIVE_BAYES = "naive_bayes"
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_VOTING = "wighted_voting"
    AUTO = "auto"


class Metric:
    CONFUSION_MATRIX = "confusion_matrix"
    ACCURACY = "accuracy"


class Native:
    PREDICTIONS = "Y_predictions"
    TRAIN_LABELS = "Y_test"
    # TRAIN_PREDICTIONS = "Y_train"
    TRAIN_PREDICTIONS = "Y_predictions"
