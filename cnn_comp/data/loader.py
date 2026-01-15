"""
    Data loader module:
    Shuffles indices and yield mini-batches.
"""
import random

class DataLoader:
    """
    Data Loader for batching and shuffling data.
    """
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.indices = list(range(self.num_samples))
        self.current_idx = 0
        if self.shuffle:
            self._shuffle_indices()

    def _shuffle_indices(self):
        random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            self._shuffle_indices()
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration

        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.data[i] for i in batch_indices]
        self.current_idx += self.batch_size
        return batch
