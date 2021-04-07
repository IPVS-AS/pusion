from pandas import DataFrame


class Report:
    def __init__(self, performance_matrix, instances, metrics):
        if performance_matrix.shape[0] != len(instances):
            raise TypeError("`performance_matrix` (axis 0) is not aligned with the number of instances.")
        if performance_matrix.shape[1] != len(metrics):
            raise TypeError("`performance_matrix` (axis 1) is not aligned with the number of metrics.")

        self.records = {}
        self.instance_names = self.__generate_instance_names(instances)
        self.metric_names = [metric.__name__ for metric in metrics]

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
        for i, instance_name in enumerate(instance_names):
            if instance_names.count(instance_name) > 1:
                instance_names[i] = "{:<35}".format(instance_name + " [" + str(i) + "]")
            else:
                instance_names[i] = "{:<35}".format(instance_name)
        return instance_names

    def __str__(self):
        return str(DataFrame.from_dict(data=self.records, orient='index', columns=self.metric_names))
