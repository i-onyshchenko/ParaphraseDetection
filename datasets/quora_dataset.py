class PreQuoraDataSet:
    def __init__(self, filename, type):
        self.type = type
        self.pairs = None
        self.labels = None
        self.size = 0
        self.pos_count = 0
        self.neg_count = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        f = open(filename)
        lines = f.readlines()[1:]
        parsed_lines = [line.replace('\n', '').split('\t') for line in lines]
        self.pairs = [[line[3], line[4]] for line in parsed_lines]
        self.labels = [int(line[-1]) for line in parsed_lines]
        self.size = len(self.labels)
        self.pos_count = self.labels.count(1)
        self.neg_count = self.size - self.pos_count
        # pos_indexes = filter(lambda i: self.labels[i] == 1, range(self.size))
        print("Number of {} pairs: {}, positive: {}, negative {}".format(self.type, self.size, self.pos_count, self.neg_count))
        f.close()

    @property
    def get_pairs(self):
        return self.pairs

    @property
    def get_labels(self):
        return self.labels


class QuoraDataSet:
    def __init__(self, train_filename=None, test_filename=None):
        self.train_data_set = PreQuoraDataSet(train_filename, "train") if train_filename is not None else None
        self.test_data_set = PreQuoraDataSet(test_filename, "test") if test_filename is not None else None

    @property
    def train_dataset(self):
        return self.train_data_set

    @property
    def test_dataset(self):
        return self.test_data_set
