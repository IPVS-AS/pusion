from pandas import DataFrame


class Report:
    """
    :class:`Report` is a string representation of the performance matrix retrieved by :class:`Evaluation` methods.

    :param performance_matrix: `numpy.array` of shape `(n_instances, n_metrics)`. Performance matrix containing
            performance values for each set instance row-wise and each set performance metric column-wise.
    :param instances: `list` of instances been evaluated which is aligned with the performance_matrix on axis `0`.
    :param metrics: `list` of metric functions which is aligned with the performance_matrix on axis `1`.
    """
    def __init__(self, performance_matrix, instances, metrics):
        if performance_matrix.shape[0] != len(instances):
            raise TypeError("`performance_matrix` (axis 0) is not aligned with the number of instances.")
        if performance_matrix.shape[1] != len(metrics):
            raise TypeError("`performance_matrix` (axis 1) is not aligned with the number of metrics.")

        self.records = {}
        self.instance_names = self.__generate_instance_names(instances)
        self.metric_names = [metric if type(metric) is str else metric.__name__ for metric in metrics]

        for i, pv in enumerate(performance_matrix):
            self.records[self.instance_names[i]] = pv

    @staticmethod
    def __generate_instance_names(instances):
        instance_names = []
        for instance in instances:
            if isinstance(instance, str):
                instance_names.append(instance)
            else:
                instance_names.append(type(instance).__name__)
        instance_names_ = list(instance_names)
        for i, instance_name in enumerate(instance_names):
            if instance_names_.count(instance_name) > 1:
                instance_names[i] = "{:<35}".format(instance_name + " [" + str(i) + "]")
            else:
                instance_names[i] = "{:<35}".format(instance_name)
        return instance_names

    def __str__(self):
        return str(DataFrame.from_dict(data=self.records, orient='index', columns=self.metric_names))

    def get_data_frame(self):
        return DataFrame.from_dict(data=self.records, orient='index', columns=self.metric_names)
