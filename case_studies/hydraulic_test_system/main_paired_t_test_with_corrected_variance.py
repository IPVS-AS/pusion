import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
import pylab

def paired_t_test_with_corrected_variance(a_metrics, b_metrics, level_of_significance=0.05):


    a_metrics_reshaped = np.reshape(a_metrics, np.shape(a_metrics)[0] * np.shape(a_metrics)[1])
    b_metrics_reshaped = np.reshape(b_metrics, np.shape(b_metrics)[0] * np.shape(b_metrics)[1])

    if np.shape(a_metrics) != np.shape(b_metrics):
        raise TypeError("Shapes of vectors containing the sample metrics are not equal!")

    if len(a_metrics_reshaped) != len(b_metrics_reshaped):
        raise TypeError("No evaluation performed.")

    ####################################################################################################################
    # Shapiro-Wilk test to test for normality:
    #
    # H0: sample is drawn from a normally-distributed population
    # H1: sample is drawn from a population that is NOT normally-distributed population
    sw_test_a = scipy.stats.shapiro(a_metrics_reshaped)
    sw_test_b = scipy.stats.shapiro(b_metrics_reshaped)

    sw_level_of_significance = 0.05

    #if sw_test_b.pvalue < sw_level_of_significance or sw_test_a.pvalue < sw_level_of_significance:
    #    print(sw_test_b)
    #    print(sw_test_a)
        #raise ValueError("Samples not from a normally-distributed population!")

    # Q-Q plot
    #scipy.stats.probplot(a_metrics_reshaped, dist="norm", plot=pylab)
    #pylab.show()

    #scipy.stats.probplot(b_metrics_reshaped, dist="norm", plot=pylab)
    #pylab.show()
    ####################################################################################################################


    ####################################################################################################################
    # Test for equal variance
    #
    # Bartlett test
    # H0: all input samples are from populations with equal variances
    bartlett_test = scipy.stats.bartlett(a_metrics_reshaped, b_metrics_reshaped)
    bartlett_level_of_significance = 0.05

    # Levene test
    # H0: All input samples are from populations with equal variances
    levene_test = scipy.stats.levene(a_metrics_reshaped, b_metrics_reshaped, center='median')
    levene_level_of_significane = 0.025

    #if bartlett_test.pvalue < bartlett_level_of_significance:
    #    print(bartlett_test)

    if levene_test.pvalue < levene_level_of_significane:
        print(levene_test)

    ####################################################################################################################


    ####################################################################################################################
    # paired two sided t test according to Bouckaert 2003

    # difference of metrics
    d = a_metrics - b_metrics

    r = np.shape(a_metrics)[0] # number of runs
    k = np.shape(b_metrics)[1] # number of folds

    # compute mean m
    m = 1/(k*r) * np.sum(d, axis=(0,1))
    m2 = np.sum(d, axis=(0,1)) / (k*r)

    m3 = 0
    for i in range(k):
        inner_sum = 0
        for j in range(r):
            inner_sum += d[j, i]
        m3 += 1/r * inner_sum
    m3 = 1/k * m3

    # compute variance \sigma^2
    d_minus_m_squared = (d - m)**2
    variance = np.sum(d_minus_m_squared, axis=(0,1)) / (k*r -1)

    variance2 = 0
    for i in range(k):
        inner_sum = 0
        for j in range(r):
            inner_sum += (d[j, i] - m)**2
        variance2 += inner_sum
    variance2 = variance2 / (k*r - 1)

    # t statistic
    df = k*r - 1
    t_d = m * np.sqrt((df + 1)) / np.sqrt(variance)

    # p value for the two-tailed t-test
    p_value_two_tailed_paired_t_test = 2 * scipy.stats.t.cdf(-np.abs(t_d), df, loc=0, scale=1)

    # p value for the one-tailed t-test
    p_value_one_tailed_paired_t_test = scipy.stats.t.cdf(t_d, df, loc=0, scale=1)

    if p_value_two_tailed_paired_t_test < level_of_significance:
        reject_h0 = True
    else:
        reject_h0 = False


    #########################################################################
    # Cohen's d statistic
    sigma_p = np.sqrt((np.var(a_metrics) + np.var(b_metrics)) / 2)
    d_cohen = np.abs(np.mean(a_metrics) - np.mean(b_metrics)) / sigma_p
    #########################################################################

    ret_dict = {'t_d': t_d,
                'p_value_two_tailed_paired_t_test': p_value_two_tailed_paired_t_test,
                'level_of_significance': level_of_significance,
                'reject_h0': reject_h0,
                'd_cohen': d_cohen}

    return ret_dict



if __name__ == '__main__':

    with open('final_try_smote_144classes_6sensors_11feats_cv_multiclass_performance_scores.pickle', 'rb') as handle:
        cv_results = pickle.load(handle)


    metrics = cv_results['meta_data']['scalar_eval_metrics']
    metric_names = [metric if type(metric) is str else metric.__name__ for metric in metrics]

    pos_acc = 0
    for i in range(len(metric_names)):
        if metric_names[i] == 'accuracy':
            continue
        i+=1

    # we performed a 5 x 3 CV
    relevant_runs = [0, 1, 2, 3, 4]
    original_shape_classifiers = np.shape(cv_results['cv_scalar_perf_tensor']['ensemble'])
    classifier_accs = np.full((original_shape_classifiers[0], len(relevant_runs), original_shape_classifiers[2]), np.nan)

    original_shape_combiners = np.shape(cv_results['cv_scalar_perf_tensor']['combiners'])
    combiners_accs = np.full((original_shape_combiners[0], len(relevant_runs), original_shape_combiners[2]), np.nan)

    for i_pos in range(len(relevant_runs)):
        relevant_data_classifiers = cv_results['cv_scalar_perf_tensor']['ensemble'][:, relevant_runs[i_pos], :, pos_acc]
        classifier_accs[:, i_pos, :] = relevant_data_classifiers

        relevant_data_combiners = cv_results['cv_scalar_perf_tensor']['combiners'][:, relevant_runs[i_pos], :, pos_acc]
        combiners_accs[:, i_pos, :] = relevant_data_combiners


    # get classifiers str names
    classifiers = cv_results['meta_data']['classifier_instances']
    instance_names = []
    for instance in classifiers:
        if isinstance(classifiers, str):
            instance_names.append(instance)
        else:
            instance_names.append(type(instance).__name__)
    instance_names_ = list(instance_names)
    for i, instance_name in enumerate(instance_names):
        if instance_names_.count(instance_name) > 1:
            instance_names[i] = "{:<35}".format(instance_name + " [" + str(i) + "]")
        else:
            instance_names[i] = "{:<35}".format(instance_name)

    num_classifiers = len(instance_names)

    #get combiners str names
    combiners = cv_results['meta_data']['combiner_instances']
    combiner_names = []
    for combiner_instance in combiners:
        if isinstance(combiner_instance, str):
            combiner_names.append(combiner_instance)
        else:
            combiner_names.append(type(combiner_instance).__name__)
    num_combiners = len(combiner_names)

    # get the best performing classifier in terms of acc as reference for the significance test
    index_acc_best_model = np.argmax(np.mean(classifier_accs, axis=(0, 1)))
    accs_best_model = classifier_accs[:, :, index_acc_best_model]
    best_classifier_name = instance_names[index_acc_best_model]

    # iterate over all combiners
    results_t_test_dict = dict()
    for i in range(np.shape(combiners_accs)[-1]):
        i_combiner_accs = combiners_accs[:, :, i]

        tested_classifier_name = best_classifier_name.replace(" ", "")
        tested_combiner_name = combiner_names[i]

        print(f"Test of {tested_classifier_name} and {tested_combiner_name}")
        t_test_result = paired_t_test_with_corrected_variance(a_metrics=accs_best_model, b_metrics=i_combiner_accs)
        print(t_test_result['reject_h0'])
        print("----------------------------------------------------------")

        key_name = f"test_of_{tested_classifier_name}_and_{tested_combiner_name}"

        t_test_result['i_name'] = tested_classifier_name
        t_test_result['j_name'] = tested_combiner_name

        results_t_test_dict[key_name] = t_test_result


    # iterate over all combiner combinations
    for i in range(np.shape(combiners_accs)[-1]):
        for j in range(np.shape(combiners_accs)[-1]):
            if i != j:
                i_combiner_accs = combiners_accs[:, :, i]
                i_tested_combiner_name = combiner_names[i]

                j_combiner_accs = combiners_accs[:, :, j]
                j_tested_combiner_name = combiner_names[j]

                print(f"Test of {i_tested_combiner_name} and {j_tested_combiner_name}")
                t_test_result = paired_t_test_with_corrected_variance(a_metrics=i_combiner_accs, b_metrics=j_combiner_accs)
                print(t_test_result['reject_h0'])
                print("----------------------------------------------------------")

                key_name = f"test_of_{i_tested_combiner_name}_and_{j_tested_combiner_name}"

                t_test_result['i_name'] = i_tested_combiner_name
                t_test_result['j_name'] = j_tested_combiner_name

                results_t_test_dict[key_name] = t_test_result

