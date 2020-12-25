import heapq


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

