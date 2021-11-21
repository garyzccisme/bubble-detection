class StratifiedTimeSeriesSplit:
    """
    Strategic Time Series Split for binary labels, make sure in every folds the validation set has enough positive labels.
    """
    def __init__(self, n_splits: int = 5, test_size: int = 120,
                 min_positive_ratio: float = 0.3, min_train_size: int = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_positive_ratio = min_positive_ratio
        if min_train_size is None:
            min_train_size = 10 * test_size
        self.min_train_size = min_train_size

    def split(self, y):
        sample_len = len(y)
        split_loc = sample_len - self.test_size
        folder_counts = 0
        while folder_counts < self.n_splits and split_loc >= self.min_train_size:
            if sum(y[split_loc: split_loc + self.test_size]) / self.test_size >= self.min_positive_ratio:
                yield range(split_loc), range(split_loc, split_loc + self.test_size)
                split_loc -= self.test_size
                folder_counts += 1
            else:
                split_loc -= 1
        if folder_counts < self.n_splits:
            print("There aren't enough folder splits satisfying the min_positive_ratio.")




