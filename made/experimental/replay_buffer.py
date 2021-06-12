import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size=50000):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, fhs, hs, u):
        for i in range(len(hs) - 1):
            data = ((fhs[i] - hs[i]).cpu().numpy(), (fhs[i] - hs[-1]).cpu().numpy(),
                    u.cpu().numpy())
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        inputs, targets, us = [], [], []
        for i in idxes:
            x, y, z = self._storage[i]
            inputs.append(x)
            targets.append(y)
            us.append(z)
        return np.stack(inputs, axis=0), np.stack(targets, axis=0), np.stack(us, axis=0)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

