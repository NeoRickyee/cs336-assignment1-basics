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

    def _compaction(self):
        if len(self.pq) > 50000 and len(self.pq) > 2 * len(self.entry_finder):
            self.pq = list(self.entry_finder.values())
            heapq.heapify(self.pq)

    def add(self, key, value):
        node: MaxHeapNode = MaxHeapNode(key, value)
        # when key already exist, disable it
        entry = self.entry_finder.get(key)
        if entry:
            entry.active = False
        self.entry_finder[key] = node
        heapq.heappush(self.pq, node)
        self._compaction()
    
    def pop(self):
        while self.pq:
            node = heapq.heappop(self.pq)
            if node.active:
                del self.entry_finder[node.key]
                return node.key, node.value
        return None, None

    def reduce(self, key, reduction):
        try:
            value = self.entry_finder[key].value - reduction
        except KeyError:
            assert False
        assert(value >= 0)
        if value == 0:
            self.entry_finder[key].active = False
            return
        self._update(key, value)
    
    def increase(self, key, addition):
        entry = self.entry_finder.get(key)
        if not entry:
            node: MaxHeapNode = MaxHeapNode(key, addition)
            self.entry_finder[key] = node
            heapq.heappush(self.pq, node)
            return
        else:
            value = entry.value + addition
            self._update(key, value)

    def _update(self, key, value):
        self.entry_finder[key].active = False
        node: MaxHeapNode = MaxHeapNode(key, value)
        self.entry_finder[key] = node
        heapq.heappush(self.pq, node)
        self._compaction()