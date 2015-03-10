"""OrderedList class

This class keeps its elements ordered according to their priority.
"""
from collections import defaultdict
import numbers
from bisect import insort

class PriorityList(object):

    def __init__(self):
        """
        Returns an OrderedReceivers object that externaly behaves
        like a list but it maintains the order of its elements
        according to their priority.
        """
        self.elements = defaultdict(list)
        self.is_ordered = True
        self.priorities = []
        self.size = 0
        self._cached_elements = None

    def __del__(self):
        pass

    def __iter__(self):
        """
        this method makes PriorityList class iterable
        """
        self._order_elements()
        for priority in reversed(self.priorities):  # highest priority first
            for element in self.elements[priority]:
                yield element

    def __getitem__(self, index):
        self._order_elements()
        return self._to_list()[index]

    def __delitem__(self, index):
        self._order_elements()
        if isinstance(index, numbers.Integral):
            index = int(index)
            if index < 0:
                index_range = [len(self)+index]
            else:
                index_range = [index]
        elif isinstance(index, slice):
            index_range = range(index.start or 0, index.stop, index.step or 1)
        else:
            raise ValueError('Invalid index {}'.format(index))
        current_global_offset = 0
        priority_counts = {priority : count for (priority, count) in
                           zip(self.priorities, [len(self.elements[p]) for p in self.priorities])}
        for priority in self.priorities:
            if not index_range:
                break
            priority_offset = 0
            while index_range:
                del_index = index_range[0]
                if priority_counts[priority] + current_global_offset <= del_index:
                    current_global_offset += priority_counts[priority]
                    break
                within_priority_index = del_index - (current_global_offset + priority_offset)
                self._delete(priority, within_priority_index)
                priority_offset += 1
                index_range.pop(0)

    def __len__(self):
        return self.size

    def add(self, new_element, priority=0, force_ordering=True):
        """
        adds a new item in the list.

        - ``new_element`` the element to be inserted in the PriorityList
        - ``priority`` is the priority of the element which specifies its
        order withing the List
        - ``force_ordering`` indicates whether elements should be ordered
        right now. If set to False, ordering happens on demand (lazy)
        """
        self._add_element(new_element, priority)
        if priority not in self.priorities:
            self._add_priority(priority, force_ordering)

    def index(self, element):
        return self._to_list().index(element)

    def remove(self, element):
        index = self.index(element)
        self.__delitem__(index)

    def _order_elements(self):
        if not self.is_ordered:
            self.priorities = sorted(self.priorities)
        self.is_ordered = True

    def _to_list(self):
        if self._cached_elements == None:
            self._order_elements()
            self._cached_elements = []
            for priority in self.priorities:
                self._cached_elements += self.elements[priority]
        return self._cached_elements

    def _add_element(self, element, priority):
        self.elements[priority].append(element)
        self.size += 1
        self._cached_elements = None

    def _delete(self, priority, priority_index):
        del self.elements[priority][priority_index]
        self.size -= 1
        if len(self.elements[priority]) == 0:
            self.priorities.remove(priority)
        self._cached_elements = None

    def _add_priority(self, priority, force_ordering):
        if force_ordering and self.is_ordered:
            insort(self.priorities, priority)
        elif not force_ordering:
            self.priorities.append(priority)
            self.is_ordered = False
        elif not self.is_ordered:
            self.priorities.append(priority)
            self._order_elements()
        else:
            raise AssertionError('Should never get here.')

