import sys
import getopt

from clunpy.decision_combiner import DecisionCombiner
from clunpy.constants import *
from clunpy.converter import *
from clunpy.input_output import *


def main_cli():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:p:f:o:h',
                                   ['method=', 'problem=', 'format=', 'output=', 'help'])
    except getopt.GetoptError:
        print("Error: To be specified")  # TODO
        sys.exit(2)

    method = None
    problem = None
    input_format = None
    output = None
    file_paths = []

    # Options handling
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-m', '--method'):
            l_arg = arg.lower()
            if l_arg in ('behaviour_knowledge_space', 'bks'):
                method = Method.BEHAVIOUR_KNOWLEDGE_SPACE
            elif l_arg in ('borda_count', 'bc'):
                method = Method.BORDA_COUNT
            elif l_arg in ('cos_similarity', 'cs'):
                method = Method.COS_SIMILARITY
            elif l_arg in ('decision_templates', 'dt'):
                method = Method.DECISION_TEMPLATES
            elif l_arg in ('dempster_shafer', 'ds'):
                method = Method.DEMPSTER_SHAFER
            elif l_arg in ('macro_majority_vote', 'mamv'):
                method = Method.MACRO_MAJORITY_VOTE
            elif l_arg in ('micro_majority_vote', 'mimv'):
                method = Method.MICRO_MAJORITY_VOTE
            elif l_arg in ('naive_bayes', 'nb'):
                method = Method.NAIVE_BAYES
            elif l_arg in ('simple_average', 'avg'):
                method = Method.SIMPLE_AVERAGE
            elif l_arg in ('wighted_voting', 'wv'):
                method = Method.WEIGHTED_VOTING
            else:
                raise NotImplementedError("Unknown method:", method)

        elif opt in ('-p', '--problem'):
            if arg in ('multiclass', 'mc'):
                problem = Problem.MULTI_CLASS
            elif arg in ('multilabel', 'ml'):
                problem = Problem.MULTI_LABEL

        elif opt in ('-f', '--format'):
            if arg == "native":
                input_format = Format.NATIVE
        else:
            err_usage()

    file_paths = args  # TODO file checks...

    if input_format == Format.NATIVE:
        dataset = load_native_files_as_data(file_paths)
        clunpy_data = convert_native_to_clunpy_data(dataset)
        combiner = DecisionCombiner()
        result = combiner.combine(predictions=clunpy_data['decision_outputs_tensor'],
                                  method=method,
                                  problem=problem,
                                  train_predictions=clunpy_data['train_predictions'],
                                  train_labels=clunpy_data['true_assignments'],
                                  evidence=clunpy_data['evidence'])
        output_data = {'Y_fused': result}
        dump_clunpy_fusion_data(output_data)
    else:
        raise NotImplementedError("Unknown format.")


def usage():
    usage_info = """
    Usage info: TODO
    """
    print(usage_info)


def err_usage():
    usage()
    sys.exit(2)


if __name__ == '__main__':
    main_cli()
