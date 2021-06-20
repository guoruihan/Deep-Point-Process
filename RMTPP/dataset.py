import torch
import numpy as np


class BatchIterator:
    def __init__(self, data, batch_size, device):
        self.event = data[0]  # N x w
        self.time = data[1]  # N x w
        self.last_time = data[2]  # N
        self.batch_size = batch_size
        self.device = device
        assert self.event.shape == self.time.shape
        assert self.event.shape[0] == self.last_time.shape[0]
        self.size = self.event.shape[0]

    def shuffle(self):
        perm = np.random.permutation(len(self))
        self.event[perm] = self.event
        self.time[perm] = self.time
        self.last_time[perm] = self.last_time

    def __len__(self):
        return self.event.shape[0]

    def __iter__(self):
        b = self.batch_size
        d = self.device
        for i in range(0, self.size, self.batch_size):
            input = (self.time[i:i+b, :-1], self.event[i:i+b, :-1], self.last_time[i:i+b])
            target = (self.time[i:i+b, -1], self.event[i:i+b, -1])
            input = (torch.tensor(input[0], device=d, dtype=torch.float32),
                     torch.tensor(input[1], device=d, dtype=torch.int64),
                     torch.tensor(input[2], device=d, dtype=torch.float32))
            target = (torch.tensor(target[0], device=d, dtype=torch.float32),
                      torch.tensor(target[1], device=d, dtype=torch.int64))
            yield input, target


class ATMDataset:
    def __init__(self, dir, device):
        self.device = device
        self._train_data = self._load_data(f'{dir}/train_event.txt', f'{dir}/train_time.txt')
        self._valid_data = self._load_data(f'{dir}/valid_event.txt', f'{dir}/valid_time.txt')
        self._test_data = self._load_data(f'{dir}/test_event.txt', f'{dir}/test_time.txt')

    @staticmethod
    def _load_data(event_fname, time_fname):
        with open(event_fname, 'r') as event_file:
            event_sequences = np.array([[int(e) for e in line.split(sep=',')] for line in event_file.readlines()], dtype=np.int32)  # N x w
        with open(time_fname, 'r') as time_file:
            time_sequences = np.array([[float(e) for e in line.split(sep=',')] for line in time_file.readlines()])  # N x w
            time_sequences = np.concatenate((np.zeros_like(time_sequences[:, 0:1]), np.diff(time_sequences, n=1, axis=1)), axis=1)  # N x w
            last_time = time_sequences[:, -2]  # N
        return event_sequences, time_sequences, last_time

    def train_iter(self, batch_size):
        return BatchIterator(self._train_data, batch_size, self.device)

    def valid_iter(self, batch_size):
        return BatchIterator(self._valid_data, batch_size, self.device)

    def test_iter(self, batch_size):
        return BatchIterator(self._test_data, batch_size, self.device)