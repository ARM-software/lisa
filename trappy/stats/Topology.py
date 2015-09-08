#    Copyright 2015-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""A Topology can be defined as an arrangement of
fundamental nodes, in various levels. Each topology
has a default level "all" which has each node represented
as a group. For example:

    +--------+---------------------------------------+
    | level  |     groups                            |
    +========+=======================================+
    | system | :code:`[[0, 1, 2, 3, 4]]`             |
    +--------+---------------------------------------+
    | cluster| :code:`[[0, 1], [2, 3, 4]]`           |
    +--------+---------------------------------------+
    | Node   | :code:`[[0], [1], [2], [3], [4], [5]]`|
    +--------+---------------------------------------+

"""

class Topology(object):
    """Topology object allows grouping of
    pivot values (called nodes) at multiple levels.
    The implementation is targeted towards CPU topologies
    but can be used generically as well

    :param clusters: clusters can be defined as a
        list of groups which are again lists of nodes.

        .. note::

            This is not a mandatory
            argument but can be used to quickly create typical
            CPU topologies.

        For Example:
        ::

            from trappy.stats.Topology import Topology

            CLUSTER_A = [0, 1]
            CLUTSER_B = [2, 3]

            clusters = [CLUSTER_A, CLUSTER_B]
            topology = Topology(clusters=clusters)

    :type clusters: list
    """

    def __init__(self, clusters=[]):
        self._levels = {}
        self._nodes = set()

        if len(clusters):
            self.add_to_level("cluster", clusters)
            cpu_level = []
            for node in self.flatten():
                cpu_level.append([node])
            self.add_to_level("cpu", cpu_level)


    def add_to_level(self, level_name, level_vals):
        """Add a group to a level

        This function allows to append a
        group of nodes to a level. If the level
        does not exist a new level is created

        :param level_name: The name of the level
        :type level_name: str

        :level_vals: groups containing nodes
        :type level_vals: list of lists:
        """

        if level_name not in self._levels:
            self._levels[level_name] = []

        self._levels[level_name] += level_vals

        for group in level_vals:
            self._nodes = self._nodes.union(set(group))

    def get_level(self, level_name):
        """Returns the groups of nodes associated
        with a level

        :param level_name: The name of the level
        :type level_name: str
        """

        if level_name == "all":
            return [self.flatten()]
        else:
            return self._levels[level_name]

    def get_index(self, level, node):
        """Return the index of the node in the
        level's list of nodes

        :param level: The name of the level
        :type level_name: str

        :param node: The group for which the inde
            is required

            .. todo::

                Change name of the arg to group

        :type node: list
        """

        nodes = self.get_level(level)
        return nodes.index(node)

    def get_node(self, level, index):
        """Get the group at the index in
        the level

        :param level: The name of the level
        :type level_name: str

        :param index: Index of the group in
            the list
        :type index: int
        """

        nodes = self.get_level(level)
        return nodes[index]

    def __iter__(self):
        return self._levels.__iter__()

    def flatten(self):
        """Return a flattened list of nodes in the
        topology
        """
        return list(self._nodes)

    def level_span(self, level):
        """Return the number of groups in a level

        :param level: The name of the level
        :type level_name: str
        """
        if level == "all":
            return len(self._nodes)
        else:
            return len(self._levels[level])

    def has_level(self, level):
        """Returns true if level is present

        :param level: The name of the level
        :type level_name: str
        """

        return (level in self._levels)

