# see https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems for data set description

# 1: Cooler condition / %:
# 3: close to total failure
# 20: reduced effifiency
# 100: full efficiency
import numpy as np

cooler_condition_labels = {
    'cooler_condition_close_to_total_failure': 3,
    'cooler_condition_reduced_efficiency': 20,
    'cooler_condition_normal': 100
}

# 2: Valve condition / %:
# 100: optimal switching behavior
# 90: small lag
# 80: severe lag
# 73: close to total failure
valve_condition_labels = {
    'valve_condition_normal': 100,
    'valve_condition_small_lag': 90,
    'valve_condition_severe_lag': 80,
    'valve_condition_close_to_total_failure': 73
}

# 3: Internal pump leakage:
# 0: no leakage
# 1: weak leakage
# 2: severe leakage
internal_pump_leakage_labels = {
    'internal_pump_leakage_no_leakage': 0,
    'internal_pump_leakage_weak_leakage': 1,
    'internal_pump_leakage_severe_leakage': 2
}

# 4: Hydraulic accumulator / bar:
# 130: optimal pressure
# 115: slightly reduced pressure
# 100: severely reduced pressure
# 90: close to total failure
hydraulic_accumulator_labels = {
    'hydraulic_accumulator_optimal_pressure': 130,
    'hydraulic_accumulator_slightly_reduced_pressure': 115,
    'hydraulic_accumulator_severely_reduced_pressure': 100,
    'hydraulic_accumulator_close_to_total_failure': 90
}

# NOT Relevant
# 5: stable flag:
# 0: conditions were stable
# 1: static conditions might not have been reached yet


normal_class_lm1 = ['cooler_condition_normal', 'valve_condition_normal', 'internal_pump_leakage_no_leakage',
                    'hydraulic_accumulator_optimal_pressure']
cooler_condition_fault_lm1 = ['cooler_condition_close_to_total_failure', 'cooler_condition_reduced_efficiency']
valve_fault_lm1 = ['valve_condition_small_lag', 'valve_condition_severe_lag', 'valve_condition_close_to_total_failure']
internal_pump_fault_lm1 = ['internal_pump_leakage_weak_leakage', 'internal_pump_leakage_severe_leakage']
hydraulic_accumulator_fault_lm1 = ['hydraulic_accumulator_slightly_reduced_pressure',
                                   'hydraulic_accumulator_severely_reduced_pressure',
                                   'hydraulic_accumulator_close_to_total_failure']

# Label mapping lm1 --> Multi-class (one hot encoded label vector)
label_mappings_lm1 = {
    'normal_class_lm1': normal_class_lm1,
    'cooler_condition_fault_lm1': cooler_condition_fault_lm1,
    'valve_fault_lm1': valve_fault_lm1,
    'internal_pump_fault_lm1': internal_pump_fault_lm1,
    'hydraulic_accumulator_fault_lm1': hydraulic_accumulator_fault_lm1
}

# 16 classes
class_names_lm1 = ['normal_class_lm1', 'cooler_condition_fault_lm1', 'valve_fault_lm1', 'internal_pump_fault_lm1',
                   'hydraulic_accumulator_fault_lm1',
                   'cooler_condition_fault_lm1_valve_fault_lm1',
                   'cooler_condition_fault_lm1_internal_pump_fault_lm1',
                   'cooler_condition_fault_lm1_hydraulic_accumulator_fault_lm1',
                   'valve_fault_lm1_internal_pump_fault_lm1', 'valve_fault_lm1_hydraulic_accumulator_fault_lm1',
                   'internal_pump_fault_lm1_hydraulic_accumulator_fault_lm1',
                   'cooler_condition_fault_lm1_valve_fault_lm1_internal_pump_fault_lm1',
                   'cooler_condition_fault_lm1_valve_fault_lm1_hydraulic_accumulator_fault_lm1',
                   'cooler_condition_fault_lm1_internal_pump_fault_lm1_hydraulic_accumulator_fault_lm1',
                   'valve_fault_lm1_internal_pump_fault_lm1_hydraulic_accumulator_fault_lm1',
                   'cooler_condition_fault_lm1_valve_fault_lm1_internal_pump_fault_lm1_hydraulic_accumulator_fault_lm1']

# Label mapping lm2 --> Multi-class (one hot encoded label vector)
label_mappings_lm2 = {
    'normal_class_lm2': normal_class_lm1,
    'cooler_condition_close_to_total_failure': 'cooler_condition_close_to_total_failure',
    'cooler_condition_reduced_efficiency': 'cooler_condition_reduced_efficiency',
    'valve_condition_small_lag': 'valve_condition_small_lag',
    'valve_condition_severe_lag': 'valve_condition_severe_lag',
    'valve_condition_close_to_total_failure': 'valve_condition_close_to_total_failure',
    'internal_pump_leakage_weak_leakage': 'internal_pump_leakage_weak_leakage',
    'internal_pump_leakage_severe_leakage': 'internal_pump_leakage_severe_leakage',
    'hydraulic_accumulator_slightly_reduced_pressure': 'hydraulic_accumulator_slightly_reduced_pressure',
    'hydraulic_accumulator_severely_reduced_pressure': 'hydraulic_accumulator_severely_reduced_pressure',
    'hydraulic_accumulator_close_to_total_failure': 'hydraulic_accumulator_close_to_total_failure'
}

# label_mappings_lm3 --> Multi-label (multi one hot encoded label vector)
label_mappings_lm3 = {
    'normal_class_lm3': normal_class_lm1,
    'cooler_condition_fault_lm3': cooler_condition_fault_lm1,
    'valve_fault_lm3': valve_fault_lm1,
    'internal_pump_fault_lm3': internal_pump_fault_lm1,
    'hydraulic_accumulator_fault_lm3': hydraulic_accumulator_fault_lm1
}

# 5 classes
class_names_lm3 = ['normal_class_lm3', 'cooler_condition_fault_lm3', 'valve_fault_lm3', 'internal_pump_fault_lm3',
                   'hydraulic_accumulator_fault_lm3']

# label_mappings_lm4 --> Multi-label (multi one hot encoded label vector)
label_mappings_lm4 = {
    'normal_class_lm4': normal_class_lm1,
    'cooler_condition_close_to_total_failure': 'cooler_condition_close_to_total_failure',
    'cooler_condition_reduced_efficiency': 'cooler_condition_reduced_efficiency',
    'valve_condition_small_lag': 'valve_condition_small_lag',
    'valve_condition_severe_lag': 'valve_condition_severe_lag',
    'valve_condition_close_to_total_failure': 'valve_condition_close_to_total_failure',
    'internal_pump_leakage_weak_leakage': 'internal_pump_leakage_weak_leakage',
    'internal_pump_leakage_severe_leakage': 'internal_pump_leakage_severe_leakage',
    'hydraulic_accumulator_slightly_reduced_pressure': 'hydraulic_accumulator_slightly_reduced_pressure',
    'hydraulic_accumulator_severely_reduced_pressure': 'hydraulic_accumulator_severely_reduced_pressure',
    'hydraulic_accumulator_close_to_total_failure': 'hydraulic_accumulator_close_to_total_failure'
}

# 11 classes
class_names_lm4 = ['normal_class_lm4',
                   'cooler_condition_close_to_total_failure',
                   'cooler_condition_reduced_efficiency',
                   'valve_condition_small_lag',
                   'valve_condition_severe_lag',
                   'valve_condition_close_to_total_failure',
                   'internal_pump_leakage_weak_leakage',
                   'internal_pump_leakage_severe_leakage',
                   'hydraulic_accumulator_slightly_reduced_pressure',
                   'hydraulic_accumulator_severely_reduced_pressure',
                   'hydraulic_accumulator_close_to_total_failure'
                   ]


# get label_vector and class_names according to lm1
def getLabelMappingsLM1(profile):
    # one row for each sample in profile, one column for each class of lm1
    label_vector = np.zeros((profile.shape[0], 16), dtype=int)

    # get label for each row and set that class = 1
    for row in range(0, profile.shape[0]):
        cooler_condition, valve_condition, pump_condition, accumulator_condition = getConditionLabels(profile[row][0],
                                                                                                      profile[row][1],
                                                                                                      profile[row][2],
                                                                                                      profile[row][3])
        row_label = getComposedLabelForLM1(cooler_condition, valve_condition, pump_condition, accumulator_condition)
        label_index = np.where(np.array(class_names_lm1) == row_label)[0][0]
        label_vector[row][label_index] = 1

    return label_vector, class_names_lm1


# get label_vector and class_names according to lm2
def getLabelMappingsLM2(profile):
    # one row for each sample in profile, one column for each class of lm2
    label_vector = np.zeros((profile.shape[0], 144), dtype=int)

    # generates all 144 class names for lm2
    class_names_lm2 = generateComposedClassNamesLM2()
    # get label for each row and set that class = 1
    for row in range(0, profile.shape[0]):
        cooler_condition, valve_condition, pump_condition, accumulator_condition = getConditionLabels(profile[row][0],
                                                                                                      profile[row][1],
                                                                                                      profile[row][2],
                                                                                                      profile[row][3])
        row_label = getComposedLabelForLM2(cooler_condition, valve_condition, pump_condition, accumulator_condition)
        label_index = np.where(np.array(class_names_lm2) == row_label)[0][0]
        label_vector[row][label_index] = 1

    return label_vector, class_names_lm2


# get label_vector and class_names according to lm3
def getLabelMappingsLM3(profile):
    # one row for each sample in profile, one column for each class of lm3
    label_vector = np.zeros((profile.shape[0], 5), dtype=int)
    # get label for each sample and set that class = 1
    for row in range(0, profile.shape[0]):
        cooler_condition, valve_condition, pump_condition, accumulator_condition = getConditionLabels(profile[row][0],
                                                                                                      profile[row][1],
                                                                                                      profile[row][2],
                                                                                                      profile[row][3])
        cooler_label = ""
        valve_label = ""
        pump_label = ""
        accumulator_label = ""

        # for each part get aggregated class label from condition
        for class_name in label_mappings_lm3:
            if cooler_condition in label_mappings_lm3.get(class_name):
                cooler_label = class_name
            if valve_condition in label_mappings_lm3.get(class_name):
                valve_label = class_name
            if pump_condition in label_mappings_lm3.get(class_name):
                pump_label = class_name
            if accumulator_condition in label_mappings_lm3.get(class_name):
                accumulator_label = class_name

        # everything works normally
        if (cooler_label == 'normal_class_lm3'
                and valve_label == 'normal_class_lm3'
                and pump_label == 'normal_class_lm3'
                and accumulator_label == 'normal_class_lm3'):
            label_vector[row][0] = 1
        # get index of the fault class in the label vector + set it = 1
        else:
            if cooler_label != 'normal_class_lm3':
                cooler_index = np.where(np.array(class_names_lm3) == cooler_label)[0][0]
                label_vector[row][cooler_index] = 1
            if valve_label != 'normal_class_lm3':
                valve_index = np.where(np.array(class_names_lm3) == valve_label)[0][0]
                label_vector[row][valve_index] = 1
            if pump_label != 'normal_class_lm3':
                pump_index = np.where(np.array(class_names_lm3) == pump_label)[0][0]
                label_vector[row][pump_index] = 1
            if accumulator_label != 'normal_class_lm3':
                accumulator_index = np.where(np.array(class_names_lm3) == accumulator_label)[0][0]
                label_vector[row][accumulator_index] = 1

    return label_vector, class_names_lm3


# get label_vector and class_names according to lm4
def getLabelMappingsLM4(profile):
    # one row for each sample in profile, one column for each class of lm4
    label_vector = np.zeros((profile.shape[0], 11), dtype=int)
    # get label for each sample and set that class = 1
    for row in range(0, profile.shape[0]):
        cooler_label, valve_label, pump_label, accumulator_label = getConditionLabels(profile[row][0],
                                                                                      profile[row][1],
                                                                                      profile[row][2],
                                                                                      profile[row][3])
        # everything works normally
        if (cooler_label in normal_class_lm1
                and valve_label in normal_class_lm1
                and pump_label in normal_class_lm1
                and accumulator_label in normal_class_lm1):
            label_vector[row][0] = 1
        # get index of the fault class in the label vector + set it = 1
        else:
            if cooler_label not in normal_class_lm1:
                cooler_index = np.where(np.array(class_names_lm4) == cooler_label)[0][0]
                label_vector[row][cooler_index] = 1
            if valve_label not in normal_class_lm1:
                valve_index = np.where(np.array(class_names_lm4) == valve_label)[0][0]
                label_vector[row][valve_index] = 1
            if pump_label not in normal_class_lm1:
                pump_index = np.where(np.array(class_names_lm4) == pump_label)[0][0]
                label_vector[row][pump_index] = 1
            if accumulator_label not in normal_class_lm1:
                accumulator_index = np.where(np.array(class_names_lm4) == accumulator_label)[0][0]
                label_vector[row][accumulator_index] = 1

    return label_vector, class_names_lm4


# takes the number and returns the corresponding key for each part
def getConditionLabels(cooler_no, valve_no, pump_no, accumulator_no):
    return list(cooler_condition_labels.keys())[
               list(cooler_condition_labels.values()).index(cooler_no)], list(valve_condition_labels.keys())[
               list(valve_condition_labels.values()).index(valve_no)], list(internal_pump_leakage_labels.keys())[
               list(internal_pump_leakage_labels.values()).index(pump_no)], list(hydraulic_accumulator_labels.keys())[
               list(hydraulic_accumulator_labels.values()).index(accumulator_no)]


# returns the label for one row of profile according to the lm1 mapping
def getComposedLabelForLM1(cooler_cond, valve_cond, pump_cond, accumulator_cond):
    label = ""
    label_mapping = label_mappings_lm1

    # everything works normally
    if cooler_cond in normal_class_lm1 \
            and valve_cond in normal_class_lm1 \
            and pump_cond in normal_class_lm1 \
            and accumulator_cond in normal_class_lm1:
        label = list(label_mapping.keys())[
            list(label_mapping.values()).index(normal_class_lm1)]
    # one or more parts have faults
    else:
        if cooler_cond in cooler_condition_fault_lm1:
            label = label + list(label_mapping.keys())[
                list(label_mapping.values()).index(cooler_condition_fault_lm1)]
        if valve_cond in valve_fault_lm1:
            label = label + "_" + list(label_mapping.keys())[
                list(label_mapping.values()).index(valve_fault_lm1)]
        if pump_cond in internal_pump_fault_lm1:
            label = label + "_" + list(label_mapping.keys())[
                list(label_mapping.values()).index(internal_pump_fault_lm1)]
        if accumulator_cond in hydraulic_accumulator_fault_lm1:
            label = label + "_" + list(label_mapping.keys())[
                list(label_mapping.values()).index(hydraulic_accumulator_fault_lm1)]
        if "_" == label[0]:
            label = label[1:]

    return label


# generates all 144 labels for lm2
def generateComposedClassNamesLM2():
    # classes of 1 fault
    label = ['cooler_condition_close_to_total_failure',
             'cooler_condition_reduced_efficiency',
             'valve_condition_small_lag',
             'valve_condition_severe_lag',
             'valve_condition_close_to_total_failure',
             'internal_pump_leakage_weak_leakage',
             'internal_pump_leakage_severe_leakage',
             'hydraulic_accumulator_slightly_reduced_pressure',
             'hydraulic_accumulator_severely_reduced_pressure',
             'hydraulic_accumulator_close_to_total_failure']
    # classes of 2 faults
    label2a = []
    label2b = []
    label2c = []
    for i in range(0, 2):  # 1
        for j in range(2, 10):  # 2,3,4
            string = label[i] + "_" + label[j]
            label2a = label2a + [string]
    for i in range(2, 5):  # 2
        for j in range(5, 10):  # 3,4
            string = label[i] + "_" + label[j]
            label2b = label2b + [string]
    for i in range(5, 7):  # 3
        for j in range(7, 10):  # 4
            string = label[i] + "_" + label[j]
            label2c = label2c + [string]
    # classes of 3 faults
    label3a = []
    label3b = []
    label3c = []
    for i in range(0, 2):  # 1
        for j in range(2, 5):  # 2
            for k in range(5, 10):  # 3,4
                string = label[i] + "_" + label[j] + "_" + label[k]
                label3a = label3a + [string]
    for i in range(0, 2):  # 1
        for j in range(5, 7):  # 3
            for k in range(7, 10):  # 4
                string = label[i] + "_" + label[j] + "_" + label[k]
                label3b = label3b + [string]
    for i in range(2, 5):  # 2
        for j in range(5, 7):  # 3
            for k in range(7, 10):  # 4
                string = label[i] + "_" + label[j] + "_" + label[k]
                label3c = label3c + [string]
    # classes of 4 faults
    label4a = []
    for i in range(0, 2):  # 1
        for j in range(2, 5):  # 2
            for k in range(5, 7):  # 3
                for l in range(7, 10):  # 4
                    string = label[i] + "_" + label[j] + "_" + label[k] + "_" + label[l]
                    label4a = label4a + [string]

    return ['normal_class_lm2'] + label + label2a + label2b + label2c + label3a + label3b + label3c + label4a


# returns the label for one row of profile according to the lm2 mapping
def getComposedLabelForLM2(cooler_cond, valve_cond, pump_cond, accumulator_cond):
    label = ""
    label_mapping = label_mappings_lm2

    # everything works normally
    if cooler_cond in normal_class_lm1 \
            and valve_cond in normal_class_lm1 \
            and pump_cond in normal_class_lm1 \
            and accumulator_cond in normal_class_lm1:
        label = list(label_mapping.keys())[
            list(label_mapping.values()).index(normal_class_lm1)]
    # one or more parts have faults
    else:
        # determine cooler fault
        if cooler_cond not in normal_class_lm1:
            label = label + list(label_mapping.keys())[
                list(label_mapping.values()).index(cooler_cond)]
        # determine valve fault
        if valve_cond not in normal_class_lm1:
            label = label + "_" + list(label_mapping.keys())[
                list(label_mapping.values()).index(valve_cond)]
        # determine pump fault
        if pump_cond not in normal_class_lm1:
            label = label + "_" + list(label_mapping.keys())[
                list(label_mapping.values()).index(pump_cond)]
        # determine hydraulic accumulator fault
        if accumulator_cond not in normal_class_lm1:
            label = label + "_" + list(label_mapping.keys())[
                list(label_mapping.values()).index(accumulator_cond)]
        if "_" == label[0]:
            label = label[1:]

    return label
