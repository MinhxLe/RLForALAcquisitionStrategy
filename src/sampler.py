"""
LIFTED FROM https://dagshub.com/ActuallyOpenAI/goingBALD
"""
import heapq
import numpy as np
import random

from src.utils.utils import (
    batch_sample_slices,
    batch_sample_indices,
)
from typing import List
import numpy as np
import tensorflow as tf


class ActiveLearningSamplerT:
    """
    ActiveLearningSampler manages a index dataset
    and what is labeled/unlabled
    """
    def __init__(self, n_elements):
        self.labelled_idx_set = set()
        self.unlabelled_idx_set = set([i for i in range(n_elements)])

    @property
    def n_labelled(self):
        return len(self.labelled_idx_set)

    def label_n_elements(self, n_elements: int, **kwargs) -> int:
        """
        chooses n labeled indices to labeled
        returns # of new elemnts labelled
        """
        # labels
        assert NotADirectoryError("not implemented")

    def get_labelled_set(self):
        return self.labelled_idx_set


class ALRandomSampler(ActiveLearningSamplerT):

    def label_n_elements(self, n_elements: int) -> int:
        n_sampled = min(len(self.unlabelled_idx_set), n_elements)
        new_labels = set(random.sample(self.unlabelled_idx_set, n_sampled))
        self.labelled_idx_set |= new_labels
        self.unlabelled_idx_set -= new_labels
        return n_sampled


class FixedHeap:
    def __init__(self, key=lambda x:x):
        # https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate
        self.key = key
        self._heap = []
        self.index = 0

    def __len__(self):
        return len(self._heap)

    def data_to_heap_data(self, data):
        return (self.key(data), self.index, data)

    def push(self, data):
        heapq.heappush(self._heap, self.data_to_heap_data(data))
        self.index += 1

    def top(self):
        return self._heap[0][2]

    def pop(self):
        return heapq.heappop(self._heap)[2]


class LeastConfidenceSampler(ActiveLearningSamplerT):
    _batch_sampler_size = 32

    def __init__(self, train_data):
        n_elements = len(train_data)
        super().__init__(n_elements)
        self.train_data = train_data

    def label_n_elements(
            self,
            n_elements: int,
            model,
            ) -> int:
        """
        chooses n labeled indices to labeled
        returns # of new elemnts labelled
        """
        n_to_sample = min(len(self.unlabelled_idx_set), n_elements)
        unlabelled_indices = list(self.unlabelled_idx_set)
        heap = FixedHeap(key=lambda x : x[0])

        train_x = self.train_data
        # we need to keep the original indices
        for batch_indices in batch_sample_slices(unlabelled_indices, shuffle=False):
            batch_x = train_x[batch_indices]
            prediction = model(batch_x, training=False)

            # we get absolute value of prediction logit which is how confident
            confidences = tf.math.abs(prediction)

            for confidence, index in zip(confidences, batch_indices):
                if len(heap) < n_to_sample:
                    # push - confidnece since we want to pop most confident
                    heap.push((-confidence, index))
                else:
                    top_confidence, _ = heap.top()
                    if confidence < -top_confidence:
                        heap.pop()
                        heap.push((-confidence, index))
        while len(heap) > 0:
            _, idx = heap.pop()
            self.labelled_idx_set.add(idx)
            self.unlabelled_idx_set.remove(idx)
        del heap
        return n_to_sample