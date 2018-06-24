import numpy as np


class PredictResult:
    def __init__(self):
        self.arr_type = np.dtype([('class', 'U30'), ('probability', 'f8')])
        self.values = np.array([], dtype=self.arr_type)
        self.finished = False
        self.max_entries = 10

    def append(self, class_name, probability):
        self.finished = False
        if len(class_name) > 30:
            print('Name {0} is longer than the maximum of 30 and will be trimmed.'.format(class_name))
        self.values = np.append(self.values, np.array((class_name, probability), dtype=self.arr_type))
        if len(self.values) > self.max_entries:
            self.finish()
            self.values = self.values[0:self.max_entries]

    def finish(self):
        assert not self.finished
        self.finished = True
        self.values = np.sort(self.values, order='probability')

    def get(self):
        if not self.finished:
            self.finish()
        return self.values

    @staticmethod
    def name(item) -> str:
        return item[0]

    @staticmethod
    def difference(item) -> float:
        return item[1]
