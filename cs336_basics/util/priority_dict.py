import heapq

class MaxHeapNode:
    __slots__ = ['value', 'key', 'active']

    def __init__(self, key, value):
        self.value = value
        self.key = key
        self.active = True
    
    def __lt__(self, other):
        # largest value at top of heap
        if self.value != other.value:
            return self.value > other.value
        # equal value, largest key at top of heap
        return self.key > other.key

class PriorityDict:
    def __init__(self):
        self.entry_finder = dict()
        self.pq = []

    def add(self, key, value):
        # when key already exist, disable it
        if key in self.entry_finder:
            self.entry_finder[key].active = False

        node: MaxHeapNode = MaxHeapNode(key, value)
        self.entry_finder[key] = node
        heapq.heappush(self.pq, node)
    
    def pop(self):
        while self.pq:
            node = heapq.heappop(self.pq)
            if node.active:
                del self.entry_finder[node.key]
                return node.key, node.value
        return None, None