from mat_import import load_two_parts_of_mat
from label_mappings import *
from preprocess_data import *
from model_training_and_evaluation import *
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import os
import pickle

def main():
    feats_multi_class_unique, profile = load_two_parts_of_mat("featsForFusion6Sensors11Feats.mat",
                                                              "featsMultiClassUnique", "profile")
    #label_vector_lm1, class_names_lm1 = getLabelMappingsLM1(profile)
    label_vector_lm1, class_names_lm1 = getLabelMappingsLM2(profile)
    #label_vector_lm3, class_names_lm3 = getLabelMappingsLM3(profile)
    #label_vector_lm4, class_names_lm4 = getLabelMappingsLM4(profile)

    b_using_resampling = 2
    if b_using_resampling == 1:
        #standard oversampling with bootstrap
        argmax_class_names_lm1 = np.argmax(label_vector_lm1, axis=1)
        print(Counter(argmax_class_names_lm1))
        oversampler = RandomOverSampler(sampling_strategy={0: 60})
        feats_multi_class_unique, argmax_class_names_lm1 = oversampler.fit_resample(feats_multi_class_unique, argmax_class_names_lm1)
        print(Counter(argmax_class_names_lm1))
        label_vector_lm1 = np.zeros((len(argmax_class_names_lm1), np.max(argmax_class_names_lm1) + 1))
        label_vector_lm1[list(range(len(argmax_class_names_lm1))), list(argmax_class_names_lm1)] = int(1)

    elif b_using_resampling == 2:
        # SMOTE - synthetic minority over-sampling
        argmax_class_names_lm1 = np.argmax(label_vector_lm1, axis=1)
        print(Counter(argmax_class_names_lm1))
        oversampler = SMOTE(sampling_strategy={0: 40}, k_neighbors=3, )
        feats_multi_class_unique, argmax_class_names_lm1 = oversampler.fit_resample(feats_multi_class_unique,
                                                                                    argmax_class_names_lm1)
        print(Counter(argmax_class_names_lm1))
        label_vector_lm1 = np.zeros((len(argmax_class_names_lm1), np.max(argmax_class_names_lm1)+1))
        label_vector_lm1[list(range(len(argmax_class_names_lm1))), list(argmax_class_names_lm1)] = int(1)

    num_iterations = 5

    for num_iteration in range(num_iterations):
        X_folds_lm1, Y_folds_lm1 = preprocess_data_kfold_cross_validation(data=feats_multi_class_unique,
                                                                          labels=label_vector_lm1,
                                                                          n_splits=3,
                                                                          transformation_type="normalization",
                                                                          mapping_range=[-1, 1])

        # perform the k+1 fold cross validation to evaluate the fusion algorithms
        results_dict = k_plus_1_cross_validation_multi_class(X_folds=X_folds_lm1,
                                                             Y_folds=Y_folds_lm1,
                                                             parallelize=False,
                                                             continuous_out=True)

        pickle_file_name = 'try_smote_144classes_6sensors_11feats_iter_' + str(num_iteration+1) + '_results_2_1_fold_cv_lm1'
        with open('.' + os.sep + pickle_file_name + '.pickle', 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finish!")


if __name__ == '__main__':
    main()
