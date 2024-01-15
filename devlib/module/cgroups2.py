#    Copyright 2022 ARM Limited
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

"""

Successor to the ``cgroups`` devlib module.

This one handles both cgroups V1 and V2 transparently with an API matching
cgroup v2 semantic.

It also handles the cgroup delegation API of systemd.


.. code-block:: python

    # Necessary Imports
    from devlib import LinuxTarget
    from devlib.module.cgroups2 import RequestTree

    # Connecting to target device. Configure appropriately.

    my_target = LinuxTarget(connection_settings={
                        "host":"127.0.0.1",
                        "port":"0000",
                        "username":"root",
                        "password":"root"
                        })

    # Instantiating the RequestTree object,
    # representing a hierarchical CGroup structure consisting
    # of a singular parent and child CGroup relationship.

    request = RequestTree(
        name="root",
        children=[
            RequestTree(
                name="child",
                controllers={"pids": {"max": 10}}
            )
        ],
        controllers={"pids": {"max": 20}},
    )

    # Printing the request to display/inspect the hierarchical
    # tree-like structure of the RequestTree object.

    print(request)

    '''
    └──root (pids) {'max': 20}
        └──child (pids) {'max': 10}
    '''

    # To set-up either CGroup version hierarchies, ensure the target device is
    # appropriately configured alongside a CGroup version appropriate RequestTree object.

    # Setting up the RequestTree object CGroup hierarchy onto target device
    # as a V1 hierarchy, and printing the returned ResponseTree object.

    with request.setup_hierarchy(target=my_target, version=1) as CGroup_hierarchy:
            print(CGroup_hierarchy)

    '''
    └──root/ pids@/sys/fs/cgroup/pids/system.slice/devlib-42c838fe4f0b4f518825c4e312590113.service/root
        └──child/ pids@/sys/fs/cgroup/pids/system.slice/devlib-42c838fe4f0b4f518825c4e312590113.service/root/child
    '''

    # Setting up the RequestTree object CGroup hierarchy onto target device
    # as a V1 hierarchy, and adding a process to the 'child' CGroup.

    with request.setup_hierarchy(target=my_target, version=1) as CGroup_hierarchy:
        child = CGroup_hierarchy["child"]
        child.add_process(1234)

    # Setting up the RequestTree object CGroup hierarchy onto target device
    # as a V2 hierarchy, and adding a thread to the 'child' CGroup.

    with request.setup_hierarchy(target=my_target, version=2) as CGroup_hierarchy:
        child = CGroup_hierarchy["child"]
        child.add_thread(1234)

"""

import collections.abc
import itertools
import os
import re
import uuid
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from shlex import quote
from typing import Dict, Set, List, Union, Any
from uuid import uuid4

from devlib import LinuxTarget
from devlib.exception import (
    TargetStableCalledProcessError,
    TargetStableError,
)
from devlib.target import FstabEntry
from devlib.utils.misc import memoized


def _is_systemd_online(target: LinuxTarget):
    """
    Determines if systemd is activated on the target system.

    :param target: Interface to the target device.
    :type target: Target

    :return: Returns ``True`` if systemd is active, ``False`` otherwise.
    :rtype: bool
    """

    try:
        target.execute("systemctl status 2>&1 >/dev/null")
    except TargetStableCalledProcessError:
        return False
    else:
        return True


def _read_lines(target: LinuxTarget, path: str):
    """
    Reads the lines of a file stored on the target device.

    :param target: Interface to target device.
    :type target: Target

    :param path: The path to the file to be read.
    :type path: str

    :return: A list of the words/sentences that result from splitting
        the read file (trailing and leading white-spaces removed) delimiting on the new-line character.
    :rtype: List[str]
    """

    return target.read_value(path=path).split("\n")


def _add_controller_versions(controllers: Dict[str, Dict[str, int]]):
    """
    Finds the CGroup controller's version and adds it as a ``version`` key.

    :param controllers: A dictionary mapping ``str`` controller names to dictionaries,
        where the later dictionary contains ``hierarchy`` and ``num_cgroup`` keys mapped to their
        respective suitable ``int`` values.
    :type controllers: Dict[str, Dict[str, int]]

    :return: A dictionary mapping ``str`` controller names to dictionaries,
        where the later dictionary contains an appended ``version`` key which maps to an ``int``
        value representing the version of the respective controller if applicable.
    :rtype: Dict[str, Dict[str,int]]
    """

    # Read how the controller versions can be determined here:
    # https://man7.org/linux/man-pages/man7/cgroups.7.html
    # (Under NOTES) [Dated 12/08/2022]

    def infer_version(config):
        if config["hierarchy"] != 0:
            return 1
        elif config["hierarchy"] == 0 and config["num_cgroups"] > 1:
            return 2
        else:
            return None

    return {
        controller: {**config, "version": version} if version is not None else config
        for (controller, config, version) in (
            (controller, config, infer_version(config))
            for (controller, config) in controllers.items()
        )
    }


def _add_controller_mounts(
    controllers: Dict[str, Dict[str, int]], target_fs_list: List[FstabEntry]
):
    """
    Find the CGroup controller's mount point and adds it as ``mount_point`` key.

    :param controllers: A dictionary mapping ``str`` controller names to dictionaries,
        where the later dictionary contains `hierarchy``, ``num_cgroup`` and if appropriate ``version``
        keys mapped to their respective suitable ``int`` values.
    :type controllers: Dict[str, Dict[str, int]]

    :param target_fs_list: A list of entries of the NamedTuple type ``FstabEntry``,
        where each represents a mounted filesystem on the target device.
    :type target_fs: List[FstabEntry]

    :return: A dictionary mapping ``str`` controller names to dictionaries,
        where the later dictionary contains an appended ``mount_point`` key which maps to the suitable
        ``str`` value of the respective controllers if applicable.
    :rtype: Dict[str, Dict[str, Union[str,int]]]
    """

    # Filter the mounted filesystems on the target device, obtaining the respective V1/V2 FstabEntries.
    v1_mounts = [fs for fs in target_fs_list if fs.fs_type == "cgroup"]
    v2_mounts = [fs for fs in target_fs_list if fs.fs_type == "cgroup2"]

    def _infer_mount(controller: str, configuration: Dict):
        controller_version = configuration.get("version")
        if controller_version == 1:
            for mount in v1_mounts:
                if controller in mount.options.strip().split(","):
                    return mount.mount_point

        elif controller_version == 2:
            # If a controller is V2, a V2 hierarchy must exist. Therefore this is a legal
            # operation.
            return v2_mounts[0].mount_point

        return None

    return {
        controller: {**config, "mount_point": path if path is not None else config}
        for (controller, config, path) in (
            (
                controller,
                config,
                _infer_mount(controller=controller, configuration=config),
            )
            for (controller, config) in controllers.items()
        )
    }


def _get_cgroup_controllers(target: LinuxTarget):
    """
    Returns the CGroup controllers that are currently enabled on the target device, alongside their appropriate configurations.

    :param target: Interface to target device.
    :type target: Target

    :return: A dictionary of controller name keys to dictionary value mappings,
        where the secondary dictionary contains a mapping between various CGroup controller configuration keys
        and their respectively obtained values for the respective CGroup controllers.
    :rtype: Dict[str, Dict[str,Union[str,int]]]
    """

    # A snippet of the /proc/cgroup is shown below. The column entries are separated
    # by '\t'. The regex pattern is structured to match and group these entries.
    #
    #     #subsys_name	hierarchy	num_cgroups	enabled
    #      cpuset           3	        1	        1

    PROC_MOUNT_REGEX = re.compile(
        r"^(?!#)(?P<name>.+)\t(?P<hierarchy>.+)\t(?P<num_cgroups>.+)\t(?P<enabled>.+)"
    )

    proc_cgroup_file = _read_lines(target=target, path="/proc/cgroups")

    def _parse_controllers(controller):
        match = PROC_MOUNT_REGEX.match(controller.strip())
        if match:
            name = match.group("name")
            enabled = int(match.group("enabled"))
            hierarchy = int(match.group("hierarchy"))
            num_cgroups = int(match.group("num_cgroups"))
            # We should ignore disabled controllers.
            if enabled != 0:
                config = {
                    "hierarchy": hierarchy,
                    "num_cgroups": num_cgroups,
                }
                return (name, config)
        return (None, None)

    controllers = dict(map(_parse_controllers, proc_cgroup_file))
    controllers.pop(None)
    controllers = _add_controller_versions(controllers=controllers)
    controllers = _add_controller_mounts(
        controllers=controllers,
        target_fs_list=target.list_file_systems(),
    )

    return controllers


@contextmanager
def _request_delegation(target: LinuxTarget):
    """
    Requests systemd to delegate a subtree CGroup hierarchy to our transient service unit.

    :yield: The Main PID of the delegated transient service unit.
    :rtype: int
    """

    service_name = "devlib-" + str(uuid.uuid4().hex)

    try:
        target.execute(
            'systemd-run --no-block --property Delegate="yes" --unit {name} --quiet {busybox} sh -c "while true; do sleep 1d; done"'.format(
                name=quote(service_name), busybox=quote(target.busybox)
            ),
            as_root=True,
        )

        pid = int(
            target.execute(
                "systemctl show --property MainPID --value {name}".format(
                    name=quote(service_name)
                )
            ).strip()
        )

        yield pid

    finally:
        target.execute(
            "systemctl kill {name}".format(name=quote(service_name)), as_root=True
        )


@contextmanager
def _mount_v2_controllers(target: LinuxTarget):
    """
    Mounts the V2 unified CGroup controller hierarchy.

    :param target: Interface to target device.
    :type target: Target

    :yield: The path to the root of the mounted V2 controller hierarchy.
    :rtype: str
    
    :raises TargetStableError: Occurs in the case where the root directory of the requested CGroup V2 Controller hierarchy 
        is unable to be created up on the target system.
    """

    path = target.tempfile()
    
    try:
        target.makedirs(path, as_root=True)
    except TargetStableCalledProcessError:
        raise TargetStableError("Un-able to create the root directory of the requested CGroup V2 hierarchy")
        
        
    try:
        target.execute(
            "{busybox} mount -t cgroup2 none {path}".format(
                busybox=quote(target.busybox), path=quote(path)
            ),
            as_root=True,
        )
        yield path
    finally:
        target.execute(
            "{busybox} umount {path} && {busybox} rmdir -- {path}".format(
                busybox=quote(target.busybox),
                path=quote(path),
            ),
            as_root=True,
        )


@contextmanager
def _mount_v1_controllers(target: LinuxTarget, controllers: Set[str]):
    """
    Mounts the V1 split CGroup controller hierarchies.

    :param target: Interface to target device.
    :type target: Target

    :param controllers: The names of the CGroup controllers required to be mounted.
    :type controllers: Set[str]

    :yield: A dictionary mapping CGroup controller names to the paths that they're currently mounted at.
    :rtype: Dict[str,str]
    
    :raises TargetStableError: Occurs in the case where the root directory of a requested CGroup V1 Controller hierarchy 
        is unable to be created up on the target system.
    """

    # Internal helper function which mounts a single V1 controller hierarchy and returns
    # its mount path.
    @contextmanager
    def _mount_controller(controller):

        path = target.tempfile()
        
        try:
            target.makedirs(path, as_root=True)
        except TargetStableCalledProcessError as err:
            raise TargetStableError("Un-able to create the root directory of the {controller} CGroup V1 hierarchy".format(controller = controller))

        try:
            target.execute(
                "{busybox} mount -t cgroup -o {controller} none {path}".format(
                    busybox=quote(target.busybox),
                    controller=quote(controller),
                    path=quote(path),
                ),
            )

            yield path

        finally:
            target.execute(
                "{busybox} umount {path} && {busybox} rmdir -- {path}".format(
                    busybox=quote(target.busybox),
                    path=quote(path),
                ),
                as_root=True,
            )

    with ExitStack() as stack:
        yield {
            controller: stack.enter_context(_mount_controller(controller))
            for controller in controllers
        }


def _validate_requested_hierarchy(
    requested_controllers: Set[str], available_controllers: Dict
):
    """
    Validates that the requested hierarchy is valid using the controllers available on the target system.

    :param requested_controllers: A set of ``str``, representing the controllers that are requested to be used in the
        user defined hierarchy.
    :type requested_controllers: Set[str]

    :param available_controllers: A dictionary where the primary keys represent the available CGroup controllers on the target system.
    :type available_controllers: Dict

    :raises TargetStableError: Occurs in the case where the requested CGroup hierarchy is unable to be
        set up on the target system.
    """

    # Will determine if there are any controllers present within the requested controllers
    # and not within the available controllers

    diff = set(requested_controllers) - available_controllers.keys()

    if diff:
        raise TargetStableError(
            "Unavailable controllers: {missing}".format(missing=" ,".join(diff))
        )


class _CGroupBase(ABC):
    """
    The abstract base class that all CGroup class types' subclass.

    :param name: The name assigned to the CGroup. Used to identify the CGroup and define the CGroup directory name.
    :type name: str

    :param parent_path: The path to the parent CGroup this CGroup is a child of.
    :type parent_path: str

    :param active_controllers: A dictionary of CGroup controller name keys to dictionary value mappings,
        where the secondary dictionary contains a mapping between a specific 'attribute' of the aforementioned
        controller and a value for which that controller interface file should be set to.
    :type active_controllers: Dict[str, Dict[str, Union[str,int]]]

    :param target: Interface to target device.
    :type target: Target
    """

    def __init__(
        self,
        name: str,
        parent_path: str,
        active_controllers: Dict[str, Dict[str, str]],
        target: LinuxTarget,
    ):
        self.name = name
        self.active_controllers = active_controllers
        self.target = target
        self._parent_path = parent_path

    @property
    def group_path(self):
        return self.target.path.join(self._parent_path, self.name)

    def _set_controller_attribute(
        self, controller: str, attribute: str, value: Union[int, str], verify=False
    ):
        """
        Writes the specified ``value`` into the interface file specified by the ``controller`` and ``attribute`` parameters.
        In the case where no ``controller`` name is specified, the ``attribute`` argument is assumed to be the name of the
        interface file to write to.

        :param controller: The controller we want to select.
        :type controller: str

        :param attribute: The specific attribute of the controller we want to alter.
        :type attribute: str

        :param value: The value we want to write to the specified interface file.
        :type value: str

        :param verify: Whether we want to verify that the value is indeed written to the interface file, defaults to ``False``.
        :type verify: bool, optional
        """

        str_value = str(value)

        # Some CGroup interface files don't have a controller name prefix, we accommodate that here.
        interface_file = controller + "." + attribute if controller else attribute

        full_path = self.target.path.join(self.group_path, interface_file)

        self.target.write_value(full_path, str_value, verify=verify)

    def _create_directory(self, path: str):
        """
        Creates a new directory at the given path, creating the parent directories if required.
        If the directory already exists, no exception is thrown.

        :param path: Path to directory to be created.
        :type path: str
        """

        self.target.makedirs(path, as_root=True)

    def _delete_directory(self, path: str):
        """
        Removes the directory at the given path.

        :param path: Path to the directory to be removed.
        :type path: str
        """

        # In this context we can't use the target.remove method since that
        # tries to delete the interface/controller files as well which isn't needed nor permitted.
        self.target.execute(
            "{busybox} rmdir -- {path}".format(
                busybox=quote(self.target.busybox), path=quote(path)
            ),
            as_root=True,
        )

    def _add_process(self, pid: Union[str, int]):
        """
        Adds the process associated with the ``pid`` to the CGroup, only if
        the process is not already a member of the CGroup.

        :param pid: The PID of the process to be added to the CGroup.
        :type pid: Union[str,int]
        """

        if not self.target.file_exists(filepath="/proc/{pid}/status".format(pid=pid)):
            return TargetStableError(
                "The Process ID: {pid} does not exists.".format(pid=pid)
            )

        # The kernel disallows reading from the cgroup.procs file
        # of a threaded CGroup. When trying to add processes to
        # threaded CGroups, the process should be added to the CGroup
        # regardless. User discretion required.
        try:
            member_processes = _read_lines(
                path=self.target.path.join(self.group_path, "cgroup.procs"),
                target=self.target,
            )
        except TargetStableError:
            self._set_controller_attribute("cgroup", "procs", pid)
        
        else:
            if str(pid) not in member_processes:
                self._set_controller_attribute("cgroup", "procs", pid)

    def _get_pid_from_tid(self, tid: int):
        """
        Retrieves the ``pid`` (Process ID) that the ``tid`` (Thread ID) is a part of.

        :param tid: The Thread ID of the thread to be added to the CGroup.
        :type tid: int

        :return: The ``pid`` (Process ID) associated with the ``tid`` (Thread ID).
        :rtype: int
        """
        status = _read_lines(
            target=self.target, path="/proc/{tid}/status".format(tid=tid)
        )
        for line in status:
            # the Tgid entry contains the thread group ID, which is the PID of
            # the process this thread belongs to.
            match = re.match(r"\s*Tgid:\s*(\d+)\s*", line)
            if match:
                pid = match.group(1)
                break
        else:
            raise TargetStableError(
                "Could not get the PID of thread: {tid}".format(tid=tid)
            )

        return int(pid)

    @abstractmethod
    def _add_thread(self, tid: int, threaded_domain):
        """
        Ensures all sub-classes have the ability to add threads to their CGroups where
        their differences dont allow for a common approach.
        """
        pass

    @abstractmethod
    def _init_cgroup(self):
        """
        Ensures all sub-classes are able to initialise their respective CGroup directories
        as per defined by their user configurations.
        """
        pass

    @abstractmethod
    def __enter__(self):
        """
        Ensures all sub-classes can be used as context managers.
        """
        pass

    @abstractmethod
    def __exit__(self, *exc):
        """
        Ensures all sub-classes can be used as context managers.
        """
        pass


class _CGroupV2(_CGroupBase):
    """
    A Class representing a CGroup directory within a CGroup V2 hierarchy.

    :param name: The name assigned to the CGroup. Used to identify the CGroup and define the CGroup folder name.
    :type name: str

    :param parent_path: The path to the parent CGroup this CGroup is a child of.
    :type parent_path: str

    :param active_controllers: A dictionary of controller name keys to dictionary value mappings,
        where the secondary dictionary contains a mapping between a specific 'attribute' of the aforementioned
        controller and a value for which that controller interface file should be set to.
    :type active_controllers: Dict[str, Dict[str, Union[str,int]]]

    :param subtree_controllers:  The controllers that should be delegated to the subtree.
    :type subtree_controllers: Set[str]

    :param is_threaded: Whether the CGroup type is threaded,
        enables thread level granularity for the CGroup directory and its subtree.
    :type is_threaded: bool

    :param target: Interface to target device.
    :type target: Target
    """

    def __init__(
        self,
        name: str,
        parent_path: str,
        active_controllers: Dict[str, Dict[str, str]],
        subtree_controllers: set,
        is_threaded: bool,
        target: LinuxTarget,
    ):

        super().__init__(
            name=name,
            parent_path=parent_path,
            active_controllers=active_controllers,
            target=target,
        )
        self.subtree_controllers = subtree_controllers
        self.is_threaded = is_threaded

    def __enter__(self):
        """
        Determines what happens when we enter the context of the CGroup,
        in this case creating the required CGroup directory and calling :meth:`_init_cgroup`.
        If an exception occurs during this phase, the context will be exited and the exception raised.

        :raises TargetStableError: If an exception occurs when calling :meth:`_init_cgroup`.

        :return: An object reference to itself.
        :rtype: :class:`_CGroupV2`
        """

        self._create_directory(path=self.group_path)
        try:
            self._init_cgroup()
        except TargetStableError as err:
            self.__exit__(err, type(err), err.__traceback__)
            raise
        else:
            return self

    def __exit__(self, *exc):
        self._delete_directory(path=self.group_path)

    def _init_cgroup(self):
        """
        Performs the required steps in order to initialize the CGroup to the user specified configuration:

            * Threading the CGroup if required.
            * Write the values to be written to the specified controller interfaces files.
            * Enable and delegate the controller that the subtree requires.

        :raises TargetStableError: Occurs when domain CGroup V2 controllers have been enabled within a threaded CGroup subtree.
        """

        # Threading the CGroup if required.
        if self.is_threaded:
            # Transforming a CGroup to type 'threaded' while domain CGroup controllers
            # are enabled within the threaded subtree will result in a kernel exception.
            # As of Linux Kernel version 4.19, the following controllers
            # are threaded: cpu, perf_event, and pids.
            try:
                self._set_controller_attribute(
                    "cgroup", "type", "threaded", verify=True
                )
            except TargetStableError:
                raise TargetStableError(
                    "Domain CGroup controllers are enabled within a threaded CGroup subtree. Ensure only threaded controllers are enabled in threaded CGroups."
                )

        # Write the values to be written to the specified controller interfaces files.
        for controller, configuration in self.active_controllers.items():
            for attr, val in configuration.items():
                self._set_controller_attribute(
                    controller=controller, attribute=attr, value=val, verify=True
                )

        # Enables/Delegates the required controllers to its subtree hierarchy via cgroup.subtree_control interface file.
        for controller in self.subtree_controllers:
            self._set_controller_attribute(
                controller="cgroup",
                attribute="subtree_control",
                value="+{cont}".format(cont=controller),
            )

    def _add_thread(self, tid: int, threaded_domain):
        """
        Attempts to add the thread associated with ``tid`` to the CGroup.
        Due to the requirements imposed by the kernel regarding thread management within a V2 CGroup hierarchy,
        the process that the thread associated with ``tid`` is a part of must reside at the root of the threaded
        subtree. This method also ensures that this requirement is satisfied by migrating said process to
        the CGroup at the root of the threaded sub-tree hierarchy if required, enabling thread level granularity
        across the entire subtree.

        :param tid: The TID (Thread ID) of the thread to be added to the CGroup.
        :type tid: int

        :param threaded_domain: The :class:`ResponseTree` object representing the threaded domain
            of the threaded CGroup subtree. The process will be added to all the CGroups
            that the :class:`ResponseTree` represent.
        :type threaded_domain: :class:`ResponseTree`
        """

        pid_of_tid = self._get_pid_from_tid(tid=tid)

        for low_level in threaded_domain.low_levels.values():
            low_level._add_process(pid_of_tid)

        self._set_controller_attribute(
            controller="cgroup", attribute="threads", value=tid
        )


class _CGroupV2Root(_CGroupV2):
    """
    A subclass of the :class:`_CGroupV2` class representing a root V2 CGroup directory.
    Contains the necessary functionality that enables the setting-up / mounting of a V2
    CGroup hierarchy.

    :param mount_point: The path on which the root of the CGroup V2 hierarchy is mounted on.
    :type mount_point: str

    :param subtree_controllers:  The controllers that should be delegated to the subtree.
    :type subtree_controllers: Set[str]

    :param target: Interface to target device.
    :type target: Target
    """

    @classmethod
    def _v2_controller_translation(
        cls, controllers: Dict[str, Dict[str, Union[str, int]]]
    ):
        """
        Given the new controller names within V2, rename the controllers to provide CGroupV2 compatibility.
        At this point in time, the ``blkio`` controller has been renamed to ``io`` in V2, while the V2 ``cpu`` controller
        wraps both ``cpu`` and ``cpuacct`` controllers/sub-systems.

        :param controllers: A dictionary of controller name keys to dictionary value mappings,
            where the secondary dictionary contains a mapping between the ``version`` and `mount_point``
            keys and their respectively obtained values.
        :rtype: Dict[str, Dict[str,Union[str,int]]]

        :raises TargetStableError: In the case where the the ``cpu`` and ``cpuacct`` CGroup controllers are in use
            under different CGroup version hierarchies.

        :raises TargetStableError: In the case where either ``cpu`` / ``cpuacct`` controller is not enabled on the target device.

        :return: The amended ``controllers`` dictionary with the updated names.
        :rtype: Dict[str, Dict[str, Union[str,int]]]
        """

        translation = {}

        if "blkio" in controllers:
            translation["io"] = controllers["blkio"]

        if "cpu" in controllers and "cpuacct" in controllers:
            if controllers["cpu"].get("version") != controllers["cpuacct"].get(
                "version"
            ):
                raise TargetStableError(
                    "CPU and CPUACCT controllers differ in versions. Both required to be version 2."
                )
            else:
                translation["cpu"] = controllers["cpu"]
        else:
            raise TargetStableError(
                "Both CPU and CPUACCT controllers need to be enabled on the system to enable the V2 CPU controller."
            )

        return {
            **translation,
            **{
                controller: configuration
                for controller, configuration in controllers.items()
                # We don't to overwrite the performed controller name translation.
                if controller not in ["blkio", "cpu", "cpuacct"]
            },
        }

    @classmethod
    def _get_delegated_sub_path(cls, delegated_pid: int, target: LinuxTarget):
        """
        Returns the relative sub-path the delegated root of the V2 hierarchy is mounted on, via the parsing
        of the /proc/<PID>/cgroup file of the delegated process associated with ``delegated_pid``.

        :param delegated_pid: The Main PID of the transient service unit we requested delegation for.
        :type delegated_pid: int

        :param target: Interface to target device.
        :type target: Target

        :return: The sub-path to the delegate root of the V2 CGroup hierarchy.
        :rtype: str
        """

        relative_delegated_mount_paths = _read_lines(
            target=target, path="/proc/{pid}/cgroup".format(pid=delegated_pid)
        )

        # Following Regex matches the line that contains the relative sub path.
        REL_PATH_REGEX = re.compile(r"0::\/(?P<path>.+)")

        for mount_path in relative_delegated_mount_paths:
            m = REL_PATH_REGEX.match(mount_path)
            if m:
                return m.group("path")
            else:
                raise TargetStableError(
                    "A V2 CGroup hierarchy was not delegated by systemd."
                )

    @classmethod
    def _get_available_controllers(
        cls, controllers: Dict[str, Dict[str, Union[str, int]]]
    ):
        """
        Returns the CGroup controllers that are currently not in use on the target device,
        which can be taken control over and used in a manually mounted V2 hierarchy.
        This method will only be called in the absence of systemd.

        :param controllers: A dictionary of CGroup controller name keys to dictionary value mappings,
            where the secondary dictionary contains a mapping between various CGroup controller configuration keys
            and their respectively obtained values for the respective CGroup controllers.
        :rtype: Dict[str, Dict[str,Union[str,int]]]

        :raises TargetStableError: Occurs in the case where a V2 hierarchy is already mounted on the target device.
            We want to bail out in this case.

        :return: The ``controllers`` Dict filtered to just those controllers which are free/un-used.
        :rtype: Dict[str, Dict[str, Union[str,int]]]
        """

        # Filters the controllers dict to entries where the version is == 2.
        mounted_v2_controllers = {
            controller
            for controller, configuration in controllers.items()
            if (configuration.get("version") == 2)
        }

        if mounted_v2_controllers:
            raise TargetStableError(
                "A V2 CGroup hierarchy is already mounted on the specified target system, therefore unable to mount requested hierarchy"
            )
        else:
            return {
                controller: configuration
                for controller, configuration in controllers.items()
                if configuration.get("version") is None
            }

    @classmethod
    def _path_to_delegated_root(
        cls, controllers: Dict[str, Dict[str, Union[int, str]]], sub_path: str
    ):
        """
        Return the full path to the delegated root. This occurs in 2 stages:

            * Initially obtain the path to root mount of the unified V2 hierarchy (usually: ``/sys/fs/cgroup/path/to/root/``).
                A subtree with no controllers could be delegated (given a hybrid CGroup hierarchy),
                this is verified not be the case.

            * Creating a full path, which consists of the path concatenation of the root mount path
                and the delegated subpath.

        :param controllers: A Dictionary of currently mounted controller name keys to Dictionary value mappings,
            where the secondary dictionary contains a mapping between various CGroup controller configuration keys
            and their respectively obtained values for the respective CGroup controllers.
        :type controllers: Dict[str, Dict[str, Union[str,int]]]

        :param sub_path: The relative subpath to the delegated root hierarchy.
        :type sub_path: str

        :raises TargetStableError: Occurs in the case where no V2 controllers are active on the target.

        :return: A full path to the delegated root of the V2 CGroup hierarchy.
        :rtype: str
        """

        # Filter out non v2 controller mounts and append the "mount_point" to a set
        v2_mount_point = {
            configuration["mount_point"]
            for configuration in controllers.values()
            if configuration.get("version") == 2
        }
        if not v2_mount_point:
            raise TargetStableError(
                "No V2 CGroup controllers have been delegated on this target."
            )
        else:
            # Since there can only be a single V2 hierarchy (ignoring bind mounts), this should be totally legal.
            mount_path_to_unified_hierarchy = v2_mount_point.pop()
            return str(os.path.join(mount_path_to_unified_hierarchy, sub_path))

    @classmethod
    @contextmanager
    def _systemd_offline_mount(
        cls,
        target: LinuxTarget,
        all_controllers: Dict[str, Dict[str, Union[str, int]]],
        requested_controllers: Set[str],
    ):
        """
        Manually mounts the V2 hierarchy on the target device. Occurs in the absence of systemd.

        :param target: Interface to target device.
        :type target: Target

        :param all_controllers: A Dictionary of currently mounted controller name keys to Dictionary value mappings,
            where the secondary dictionary contains a mapping between various CGroup controller configuration keys
            and their respectively obtained values for the respective CGroup controllers.
        :type controllers: Dict[str, Dict[str, Union[str,int]]]

        :param requested_controllers: The set of controllers required to mount the requested hierarchy.
        :type requested_controllers: Set[str]

        :yield: The path to the root mount point of the unified V2 hierarchy.
        :rtype: str
        """

        unused_controllers = _CGroupV2Root._get_available_controllers(
            controllers=all_controllers
        )
        _validate_requested_hierarchy(
            requested_controllers=requested_controllers,
            available_controllers=unused_controllers,
        )

        with _mount_v2_controllers(target) as mount_path:
            yield mount_path

    @classmethod
    @contextmanager
    def _systemd_online_setup(
        cls,
        target: LinuxTarget,
        all_controllers: Dict[str, Dict[str, int]],
        requested_controllers: Set[str],
    ):
        """
        Sets up the required V2 hierarchy on the target device. Occurs in the presence of systemd.

        :param target: Interface to target device.
        :type target: Target

        :param all_controllers: A Dictionary of currently mounted CGroup controller name keys to dictionary value mappings,
            where the secondary dictionary contains a mapping between various CGroup controller configuration keys
            and their respectively obtained values for the respective CGroup controllers.
        :type all_controllers: Dict[str, Dict[str, Union[str,int]]]

        :param requested_controllers: The set of controllers required to mount the requested hierarchy.
        :type requested_controllers: Set[str]

        :yield: The path to the root of the delegated V2 CGroup hierarchy.
        :rtype: str
        """
        with _request_delegation(target=target) as main_pid:
            delegated_sub_path = _CGroupV2Root._get_delegated_sub_path(
                delegated_pid=main_pid, target=target
            )
            delegated_path = _CGroupV2Root._path_to_delegated_root(
                controllers=all_controllers,
                sub_path=delegated_sub_path,
            )

            delegated_controllers_path = "{path}/cgroup.controllers".format(
                path=delegated_path
            )

            # The controllers that have been delegated are held within
            # the 'cgroup.controllers' file. The controller names are stored on a
            # single line, requiring us to select the first (and only) element returned
            # by _read_file and splitting said element (str) using the white space character
            # as the delimiter.
            # (The _validate_requested_hierarchy requires the available_controllers argument to be a dict, necessitating this dict structure.)
            delegated_controllers = {
                controller: None
                for controller in _read_lines(
                    target=target, path=delegated_controllers_path
                )[0].split(" ")
            }

            _validate_requested_hierarchy(
                requested_controllers=requested_controllers,
                available_controllers=delegated_controllers,
            )
            yield delegated_path

    @classmethod
    @contextmanager
    def _mount_filesystem(cls, target: LinuxTarget, requested_controllers: Set[str]):
        """
        Mounts/Sets-up a V2 hierarchy on the target device, covering contexts where
        systemd is both present and absent.

        :param target: Interface to target device.
        :type target: Target

        :param requested_controllers: The set of controllers required to mount the requested hierarchy.
        :type requested_controllers: Set[str]

        :yield: A path to the root of the V2 hierarchy that has been mounted/delegated for the user.
        :rtype: str
        """

        systemd_online = _is_systemd_online(target=target)
        controllers = _CGroupV2Root._v2_controller_translation(
            _get_cgroup_controllers(target=target)
        )

        if systemd_online:
            cm = _CGroupV2Root._systemd_online_setup(
                target=target,
                all_controllers=controllers,
                requested_controllers=requested_controllers,
            )
            with cm as mount_path:
                yield mount_path

        else:
            cm = _CGroupV2Root._systemd_offline_mount(
                target=target,
                all_controllers=controllers,
                requested_controllers=requested_controllers,
            )
            with cm as mount_path:
                yield mount_path

    def __init__(
        self,
        mount_point: str,
        subtree_controllers: set,
        target: LinuxTarget,
    ):

        super().__init__(
            name="",
            parent_path=mount_point,
            # Root can not have active controllers.
            active_controllers={},
            subtree_controllers=subtree_controllers,
            # Root can not be threaded.
            is_threaded=False,
            target=target,
        )
        self.target = target

    def __enter__(self):
        """
        Determines what happens when we enter the context of the CGroup, in this case the :meth:`_init_root_cgroup` method
        is to be called; initializing the root group to abide by the user defined configuration.
        If an exception occurs during this phase, the context will be exited and the exception raised.

        :raises TargetStableError: Occurs when an exception occurs within the :meth:`_init_root_cgroup` method call.

        :return: An object reference to itself.
        :rtype: :class:`_CGroupV2Root`
        """

        try:
            self._init_root_cgroup()
        except TargetStableError as err:
            self.__exit__(err, type(err), err.__traceback__)
            raise
        else:
            return self

    def __exit__(self, *exc):
        pass

    def _init_root_cgroup(self):
        """
        Performs the required actions in order to initialise a Root V2 CGroup.
        In the case where systemd is active, there is a required need to create a leaf CGroup from the Root, where the PIDs
        systemd has delegated the subtree can be moved to. The reason for this is due to the side effect of being unable to
        change the contents of the ``cgroup.subtree_control`` interface file while processes exist within the CGroup.
        This process is skipped when initializing a root CGroup on a non-systemd system.
        """

        if _is_systemd_online(target=self.target):
            # Create the leaf CGroup directory
            group_name = "devlib-" + str(uuid4().hex)
            full_path = self.target.path.join(self.group_path, group_name)
            self._create_directory(full_path)

            delegated_pids = _read_lines(
                target=self.target,
                path="{path}/cgroup.procs".format(path=self.group_path),
            )

            # Move PIDs to leaf CGroup.
            for pid in delegated_pids:
                self.target.write_value(
                    path=self.target.path.join(full_path, "cgroup.procs"),
                    value=pid,
                    verify=False,
                )

        # Write to Subtree
        for controller in self.subtree_controllers:
            self._set_controller_attribute(
                "cgroup",
                "subtree_control",
                "+{cont}".format(cont=controller),
            )


class _CGroupV1(_CGroupBase):
    """
    A Class representing a CGroup folder within a CGroup V1 hierarchy.

    :param name: The name assigned to the CGroup. Used to identify the CGroup and define the CGroup folder name.
    :type name: str

    :param parent_path: The path to the parent CGroup this CGroup is a child of.
    :type parent_path: str

    :param active_controllers: A dictionary of controller name keys to dictionary value mappings,
        where the secondary dictionary contains a mapping between a specific 'attribute' of the aforementioned
        controller and a value for which that controller interface should be set to.

    :type active_controllers: Dict[str, Dict[str, Union[str,int]]]

    :param target: Interface to target device.
    :type target: Target
    """

    def __enter__(self):
        """
        Determines what happens when we enter the context of the CGroup,
        in this case creating the required CGroup directory and calling the :meth:`_init_cgroup` method.
        If an exception occurs during this phase, the context will be exited and the exception raised.

        :raises TargetStableError: If an exception occurs within the :meth:`_init_cgroup` method call.

        :return: An object reference to itself.
        :rtype: :class:`_CGroupV1`
        """

        self._create_directory(self.group_path)
        try:
            self._init_cgroup()
        except TargetStableError as err:
            self.__exit__(err, type(err), err.__traceback__)
            raise
        else:
            return self

    def __exit__(self, *exc):
        self._delete_directory(self.group_path)

    def _init_cgroup(self):
        """
        Performs the required steps in order to initialize the CGroup to the user specified configuration:

            * Write the values to be written to the specified controller interfaces files.
        """

        # Attributes to controller {controller : {attr : val, attr : val}}

        for controller, configuration in self.active_controllers.items():
            for attr, val in configuration.items():
                self._set_controller_attribute(
                    controller=controller, attribute=attr, value=val, verify=True
                )

    def _add_thread(self, tid: int, threaded_domain):
        """
        Adds the thread associated with ``tid`` to the CGroup.
        While thread level management suffers from no restrictions within a V1 hierarchy,
        semantic equivalence with CGroup V2 is required. Therefore, adding a thread
        to a CGroup within a V1 hierarchy still abides by the restrictions set within
        a V2 hierarchy. In this case, the process that the thread associated with ``tid``
        is a part of must reside at the root of the threaded subtree, enabling thread level
        granularity across the entire of the threaded subtree.

        :param tid: The TID of the thread to be added to the CGroup
        :type tid: int

        :param threaded_domain: The :class:`ResponseTree` object representing the threaded domain
            of the threaded CGroup subtree. The process will be added to all the CGroups
            that the :class:`ResponseTree` represents.
        :type threaded_domain: :class:`ResponseTree`
        """

        pid_of_tid = self._get_pid_from_tid(tid=tid)

        for low_level in threaded_domain.low_levels.values():
            low_level._add_process(pid_of_tid)

        self._set_controller_attribute("", "tasks", tid)


class _CGroupV1Root(_CGroupV1):
    """
    A subclass of the :class:`_CGroupV1` class representing a root V1 CGroup directory.
    Contains the necessary functionality that enables the setting-up / mounting of a V1
    CGroup hierarchy.

    :param mount_point: The path to which the root of the CGroup V1 controller hierarchy is mounted on.
    :type mount_point: str

    :param target: Interface to target device.
    :type target: Target
    """

    @classmethod
    def _get_delegated_paths(
        cls,
        controllers: Dict[str, Dict[str, Union[str, int]]],
        delegated_pid: int,
        target: LinuxTarget,
    ):
        """
        Returns the relative sub-paths the delegated roots of the V1 hierarchies, via the parsing
        of the /proc/<PID>/cgroup file of the delegated PID.

        :param controllers: A dictionary of currently mounted CGroup controller name keys to dictionary value mappings,
             where the secondary dictionary contains a mapping between various CGroup controller configuration keys
             and their respectively obtained values for the respective CGroup controllers.
        :type controllers: Dict[str, Dict[str, Union[str,int]]]

        :param delegated_pid: The Main PID of the transient service unit we request delegation for.
        :type delegated_pid: int

        :param target: Interface to target device.
        :type target: Target

        :raises TargetStableError: Occurs in the case where no V1 controllers have been delegated.

        :return: A dictionary mapping CGroup controllers to their respective delegated root paths.
        :rtype: Dict[str, str]
        """

        delegated_mount_paths = _read_lines(
            target=target, path="/proc/{pid}/cgroup".format(pid=delegated_pid)
        )

        # A snippet of the /proc/<PID>/cgroup is shown below.
        #
        # 10:misc:/
        # 9:memory:/system.slice/xyz.service
        #
        # The regex is structured to only match V1 controller hierarchies.

        REL_PATH_REGEX = re.compile(
            r"\d+:(?P<controllers>.+):\/(?P<path_to_delegated_service_root>.*)"
        )

        delegated_controllers = {}

        for mount_path in delegated_mount_paths:
            regex_match = REL_PATH_REGEX.match(mount_path)
            if regex_match:
                con = regex_match.group("controllers")
                path = regex_match.group("path_to_delegated_service_root")
                # Multiple v1 controllers can be co-mounted on a single folder hierarchy.
                co_mounted_controllers = con.strip().split(",")
                for controller in co_mounted_controllers:
                    try:
                        configuration = controllers[controller]
                    except KeyError:
                        pass
                    else:
                        delegated_controllers[controller] = target.path.join(
                            configuration["mount_point"], path
                        )

        if not delegated_controllers:
            raise TargetStableError(
                "No V1 CGroup controllers have been delegated on the target."
            )

        return delegated_controllers

    @classmethod
    @contextmanager
    def _systemd_offline_mount(
        cls,
        requested_controllers: Set[str],
        all_controllers: Dict[str, Dict[str, Union[str, int]]],
        target: LinuxTarget,
    ):
        """
        Manually mounts the V1 split hierarchy on the target device. Occurs in the absence of systemd.

        :param requested_controllers: The set of controllers required to mount the requested hierarchy.
        :type requested_controllers: Set[str]

        :param all_controllers: A Dictionary of currently mounted controller name keys to Dictionary value mappings,
            where the secondary dictionary contains a mapping between various CGroup controller configuration keys
            and their respectively obtained values for the respective CGroup controllers.
        :type all_controllers: Dict[str, Dict[str, Union[str,int]]]

        :param target: Interface to target device.
        :type target: Target

        :yield: A dictionary mapping CGroup controller names to their respective mount points.
        :rtype: Dict[str,str]
        """

        available_controllers = _CGroupV1Root._get_available_v1_controllers(
            controllers=all_controllers
        )
        _validate_requested_hierarchy(
            requested_controllers=requested_controllers,
            available_controllers=available_controllers,
        )

        cm = _mount_v1_controllers(target=target, controllers=requested_controllers)
        with cm as mounted:
            yield mounted

    @classmethod
    def _get_available_v1_controllers(
        cls, controllers: Dict[str, Dict[str, Union[int, str]]]
    ):

        unused_controllers = {
            controller: configuration
            for controller, configuration in controllers.items()
            if configuration.get("version") is None
        }

        if not unused_controllers:
            raise TargetStableError("No V1 CGroup controllers available on target.")

        return unused_controllers

    @classmethod
    @contextmanager
    def _systemd_online_setup(
        cls,
        target: LinuxTarget,
        requested_controllers: Set[str],
        all_controllers: Dict[str, Dict[str, str]],
    ):
        """
        Sets up the required V1 hierarchy on the target device. Occurs in the presence of systemd.

        :param target: Interface to target device.
        :type target: Target

        :param requested_controllers: The set of controllers required to mount the requested hierarchy.
        :type requested_controllers: Set[str]

        :param all_controllers: A Dictionary of currently mounted controller name keys to dictionary value mappings,
            where the secondary dictionary contains a mapping between various CGroup controller configuration keys
            and their respectively obtained values for the respective CGroup controllers.
        :type all_controllers: Dict[str, Dict[str, Union[str,int]]]

        :yield: A Dict[str, str] consisting of controller name keys mapped to their respective mount points.
        :rtype: Dict[str, str]
        """

        with _request_delegation(target) as pid:
            delegated_controllers = _CGroupV1Root._get_delegated_paths(
                controllers=all_controllers,
                delegated_pid=pid,
                target=target,
            )
            _validate_requested_hierarchy(
                requested_controllers=requested_controllers,
                available_controllers=delegated_controllers,
            )

            yield (delegated_controllers)

    @classmethod
    @contextmanager
    def _mount_filesystem(cls, target: LinuxTarget, requested_controllers: Set[str]):
        """
        A context manager which Mounts/Sets-up a V1 split hierarchy on the target device, covering contexts where
        systemd is both present and absent. This context manager Mounts/Sets-up a split V1 hierarchy (if possible)
        and performs the clean up (process depends on whether systemd is present or not) necessary afterwards returning
        the target device to the state before the mount/set-up occurred.

        :param target: Interface to target device.
        :type target: Target

        :param requested_controllers: The set of controllers required to mount the requested hierarchy.
        :type requested_controllers: Set[str]

        :yield: A dictionary mapping controller name to the paths where the controllers are mounted on, used to build the user requested V1 hierarchy.
        :rtype: dict[str,str]
        """

        systemd_online = _is_systemd_online(target=target)
        controllers = _get_cgroup_controllers(target=target)

        if systemd_online:
            cm = _CGroupV1Root._systemd_online_setup(
                target=target,
                requested_controllers=requested_controllers,
                all_controllers=controllers,
            )
            with cm as controllers:
                yield controllers

        else:
            cm = _CGroupV1Root._systemd_offline_mount(
                target=target,
                requested_controllers=requested_controllers,
                all_controllers=controllers,
            )
            with cm as controllers:
                yield controllers

    def __init__(self, mount_point: str, target: LinuxTarget):

        super().__init__(
            # Root name is null. Isn't required.
            name="",
            parent_path=mount_point,
            # Root can not have active controllers.
            active_controllers={},
            target=target,
        )

    def __enter__(self):
        """
        Determines what happens when we enter the context of the CGroup,
        in this case no set-up/resource-management occurs.

        :return: An object reference to itself.
        :rtype: :class:`_CGroupV1Root`
        """

        return self

    def __exit__(self, *exc):
        pass


class _TreeBase(ABC):
    """
    The abstract base class that all tree class types' subclass.

    :param name: The name assigned to the tree node.
    :type name: str

    :param is_threaded: Whether the node is threaded or not.
    :type is_threaded: bool
    """

    def __init__(self, name: str, is_threaded: bool):
        self.name = name
        self.is_threaded = is_threaded
        self.threaded_domain = self

        # Propagates Threaded Property to
        # sub-tree.
        def make_threaded(grp):
            grp.is_threaded = True
            for child in grp._children_list:
                make_threaded(child)

        # Propagates the Threaded domain
        # to sub-tree.
        def set_domain(grp):
            grp.threaded_domain = domain
            for child in grp._children_list:
                set_domain(child)

        if is_threaded:
            make_threaded(self)
        else:
            domain = self
            if any([child.is_threaded for child in self._children_list]):
                for child in self._children_list:
                    make_threaded(child)
                    set_domain(child)

    @property
    def is_threaded_domain(self):
        return (
            True
            if any([child.is_threaded for child in self._children_list])
            and self.threaded_domain is self
            else False
        )

    @property
    @memoized
    def group_type(self):
        if self.is_threaded_domain:
            return "threaded domain"
        elif self.is_threaded:
            return "threaded"
        else:
            return "domain"

    def __str__(self, level=0):
        """
        Returns a string representation of the tree hierarchy, used for visualization and debugging.

        :param level: The current depth of the tree, defaults to 0.
        :type level: int, optional

        :return: String formatted output, displaying the hierarchical structure of the tree.
        :rtype: str
        """

        TAB = "\t"
        ELBOW = "└──"
        children = "\n".join(
            child.__str__(level=level + 1) for child in (self._children_list)
        )
        children = "\n" + children if children else children

        return "{tab}{elbow}{name} {node_info} {children}".format(
            tab=TAB * level,
            elbow=ELBOW,
            name=self.name,
            node_info=self._node_information,
            children=children,
        )

    @property
    @abstractmethod
    def _node_information(self):
        """
        Returns a formatted string displaying the information the :class:`_TreeBase` object represents.
        """
        pass

    @property
    @abstractmethod
    def _children_list(self):
        """
        Returns List[:class:`_TreeBase`].
        """
        pass


class RequestTree(_TreeBase):
    """
    A class used to represent a unified, tree-like user-defined CGroup hierarchy.
    Modelled as a V2 CGroup hierarchy, but able to represent both hierarchy versions (1 & 2) on the target device as
    required by ensuring V2 semantic equivalence is maintained within the context of setting up a V1 hierarchy.

    :param name: Name assigned to the user defined :class:`RequestTree` object.
    :type name: str

    :param children: A list of :class:`RequestTree` objects representing the children the object is
        a hierarchical parent to, defaults to ``None``.
    :type children: List[:class:`RequestTree`], optional

    :param controllers: A Dictionary of controller name keys to dictionary value mappings,
        where the secondary dictionary contains a mapping between controller specific attributes and
        their respective to be assigned values, , defaults to ``None``.
    :type controllers: Dict[str, Dict[str, Union[str,int]]], optional

    :param threaded: defines whether the object will represent a CGroup capable of managing threads, defaults to ``False``.
    :type threaded: bool, optional
    """

    def __init__(
        self,
        name: str,
        children: Union[list, None] = None,
        controllers: Union[Dict[str, Dict[str, Any]], None] = None,
        threaded=False,
    ):
        self.children = children or []
        self.controllers = controllers or {}
        super().__init__(name=name, is_threaded=threaded)

    @property
    def _node_information(self):
        # Returns Requests Tree Node Information.
        active_controllers = [
            "({controller}) {config}".format(
                controller=controller, config=configuration
            )
            for controller, configuration in sorted(self.controllers.items())
        ]
        return "{active_controllers} [{group_type}]".format(
            active_controllers=", ".join(active_controllers),
            group_type=self.group_type,
        )

    @property
    @memoized
    def _all_controllers(self):
        # Returns a set of all the controllers that are active in that subtree, including its own.
        return set(
            itertools.chain(
                self.controllers.keys(),
                itertools.chain.from_iterable(
                    map(lambda child: child._all_controllers, self.children),
                ),
            )
        )

    @property
    def _subtree_controllers(self):
        # Returns a set of all the controllers that are active in that subtree, excluding its own.
        return set(
            itertools.chain.from_iterable(
                map(lambda child: child._all_controllers, self.children)
            )
        )

    @property
    def _children_list(self):
        return list(self.children)

    @contextmanager
    def setup_hierarchy(self, version: int, target: LinuxTarget):
        """
        A context manager which processes the user defined hierarchy and sets-up said hierarchy on the ``target`` device.
        Uses an internal exit stack to the handle the entering and safe exiting of the lower level
        contexts of the :class:`_CGroupBase` subclasses,
        restoring the target device to the state it was in before the hierarchy was set up.
        If the set-up was successful, it will yield an instance of the :class:`ResponseTree` class (representing the root of the tree)
        which the user will interact with and can inspect.

        :param version: The version of the CGroup hierarchy to be set up on the Target device.
        :type version: int

        :param target: Interface to target device.
        :type target: Target

        :raises TargetStableError: Occurs when the version argument is neither ``1`` or ``2``;
            the only two versions of CGroups currently available.

        :yield: An instance of the :class:`ResponseTree` class, representing the root of the CGroup hierarchy.
        """

        with ExitStack() as exit_stack:
            if version == 1:
                # Returns a {controller_name: controller_mount_point} dict
                controller_paths = exit_stack.enter_context(
                    _CGroupV1Root._mount_filesystem(
                        target=target, requested_controllers=self._all_controllers
                    )
                )
                # Mounts the Roots Controller Parents.
                root_parents = {
                    controller: _CGroupV1Root(
                        mount_point=mount_path,
                        target=target,
                    )
                    for controller, mount_path in controller_paths.items()
                    if controller in self._all_controllers
                }

                def make_groups(request: RequestTree, parents: Dict[str, _CGroupBase]):
                    """
                    Defines and instantiates the low-level :class:`_CGroupV1` objects as per defined by the
                    configuration of the ``request`` :class:`RequestTree`  object.
                    The parents of said :class:`_CGroupV1` objects will be determined via the
                    ``parents`` dictionary, ensuring the newly created :class:`_CGroupV1` objects are
                    created 'under' the suitable parent CGroup directory.

                    :param request: The :class:`RequestTree` object that'll define the required :class:`_CGroupV1` objects it represents.
                    :type request: :class:`RequestTree`

                    :param parents: The Dictionary mapping that maps CGroup controller names to their leaf CGroup directory.
                    :type parents: Dict[str, :class:`_CGroupBase`]

                    :return: A tuple ``(request_defined_cgroups, all_cgroups, parents)`` where the first element defines the
                        dictionary mapping the controller names to the :class:`_CGroupV1` objects created directly
                        due to the :class:`RequestTree` object user configuration and the second and third elements
                        defining the dictionary mapping controller names to the leaf CGroup for said controller.
                        Duplication is required not only to maintain compatibility with the function
                        defined under the same name and signature under the context of setting up a CGroup V2 hierarchy,
                        but since we want to maintain a semantic equivalence to a V2 hierarchy,
                        the low-level :class:`_CGroupV1` objects a particular :class:`RequestTree`
                        instance indirectly defines given its parents and the :class:`_CGroupV1` objects it passes to it children
                        as potential suitable parents are the same.
                    :rtype: tuple(Dict[str,:class:`_CGroupV1`], Dict[str,:class:`_CGroupV1`], Dict[str,:class:`_CGroupV1`])
                    """

                    request_defined_cgroups = {
                        controller: _CGroupV1(
                            name=request.name,
                            parent_path=parents[controller].group_path,
                            active_controllers={controller: attributes},
                            target=target,
                        )
                        for controller, attributes in request.controllers.items()
                    }

                    # Parent dict updated to include the newly created leaf CGroups.
                    parents = {**parents, **request_defined_cgroups}
                    all_cgroups = parents
                    return (request_defined_cgroups, all_cgroups, parents)

            elif version == 2:

                # Returns a string representing the root of the V2 hierarchy
                unified_mount_point = exit_stack.enter_context(
                    _CGroupV2Root._mount_filesystem(
                        target=target, requested_controllers=self._all_controllers
                    )
                )

                root_parents = _CGroupV2Root(
                    mount_point=unified_mount_point,
                    subtree_controllers=self._all_controllers,
                    target=target,
                )

                # We require to enter the context of the root of the V2 hierarchy at this stage in order to perform the required
                # root CGroup setup defined within the __enter__ method.
                exit_stack.enter_context(root_parents)

                def make_groups(request: RequestTree, parent: _CGroupV2):
                    """
                    Defines and instantiates the low-level :class:`_CGroupV2` object as per defined by the
                    configuration of the ``request`` :class:`RequestTree` object. The parents of said :class:`_CGroupV2` object
                    will be determined via the ``parents`` :class:`_CGroupV2` object, ensuring the newly created
                    :class:`_CGroupV2` object is created 'under' the suitable parent CGroup directory.

                    :param request: The :class:`RequestTree` object that'll define the required :class:`_CGroupV2` object.
                    :type request: :class:`RequestTree`

                    :param parents: The CGroup that'll be the parent of the :class:`_CGroupV2` object being defined.
                    :type parents: :class:`_CGroupV2`

                    :return: A tuple ``(controllers_to_cgroup, controllers_to_cgroup, parent)`` where the first and second elements
                        define a dictionary mapping of controller names as per defined by the :class:`RequestTree` object
                        and the solitary :class:`_CGroupV2` object that has been instantiated,
                        (All enabled controllers map to a single V2 directory under a V2 hierarchy).
                        The Last element within the tuple defines the newly instantiated :class:`_CGroupV2` object set to be the
                        hierarchical parent of the subsequent V2 CGroups to be created. Duplication is required in this case since both the paths
                        the user defined V2 controllers are enabled at and the actual paths
                        of the low-level implementation are the same as per the structure of the unified V2 hierarchy.
                    :rtype: tuple(Dict[str,:class:`_CGroupV2`],Dict[str,:class:`_CGroupV2`],:class:`_CGroupV2`)
                    """

                    request_group = _CGroupV2(
                        name=request.name,
                        parent_path=parent.group_path,
                        active_controllers=request.controllers,
                        subtree_controllers=request._subtree_controllers,
                        is_threaded=request.is_threaded,
                        target=target,
                    )

                    # Creates a mapping between the enabled controllers within this CGroup to the low-level
                    # _CGroupV2 object
                    controllers_to_cgroup = dict.fromkeys(
                        request.controllers, request_group
                    )
                    # Creating 'parent' variable for readability’s sake.
                    parent = request_group
                    return (controllers_to_cgroup, controllers_to_cgroup, parent)

            else:
                raise TargetStableError(
                    "A {version} version hierarchy cannot be mounted. Ensure requested hierarchy version is 1 or 2.".format(
                        version=version
                    )
                )

            # Create the Response Tree from the Request Tree.
            response = self._create_response(root_parents, make_groups=make_groups)
            # Returns a list of all the Low-level _CGroupBase objects the response object represents in the right order
            groups = response._all_nodes
            # Remove duplicates while preserving order.
            groups = sorted(set(groups), key=groups.index)
            # Enter the context for each object
            for group in groups:
                exit_stack.enter_context(group)

            yield response

    def _create_response(self, low_level_parent, make_groups):
        """
        Creates the :class:`ResponseTree` object tree, using the appropriately defined :meth:`make_group` callable (defined as a local function
        internally within :meth:`setup_hierarchy`) alongside the ``low_level_parent`` object to create the low-level CGroups a particular :class:`RequestTree` object represents.
        This function is then recursively called on the children of the :class:`RequestTree` object in order to create subsequent child
        :class:`ResponseTree` objects to create a Tree-like object that mirrors the Tree structure of the :class:`RequestTree` object.

        :param low_level_parent: The parent/s to the CGroups to be created. In the context of setting up a V1 hierarchy, this will be a
            dictionary mapping controller names to :class:`_CGroupV1` objects; while in the case of V2, it'll be a solitary :class:`_CGroupV2` object.
        :type low_level_parent: Dict[str,:class:`_CGroupV1`] | :class:`_CGroupV2`

        :param make_groups: The callable function definition used to create the low-level CGroup required. This callable is defined appropriately
            depending on the CGroup hierarchy version we require to set-up/mount.
        :type make_groups: callable

        :return: The root of the :class:`ResponseTree` object tree.
        :rtype: :class:`ResponseTree`
        """

        user_visible_low_level_groups, low_level_groups, low_level_parent = make_groups(
            self, low_level_parent
        )
        return ResponseTree(
            name="{name}/".format(name=self.name),
            children={
                child.name: child._create_response(
                    low_level_parent=low_level_parent,
                    make_groups=make_groups,
                )
                for child in self.children
            },
            low_levels=low_level_groups,
            # We dont want to show the user all the CGroups directories it represents within the context of a V1 hierarchy,
            # since it contains directories not directly defined by the user (The V1 hierarchy we define resembles the unified V2 hierarchy,
            # therefore there'll be some low-level _CGroupV1 objects it'll represent that haven't been directly defined by the its corresponding
            # RequestTree object but instead inherited from its parents.
            user_low_levels=user_visible_low_level_groups,
            is_threaded=self.is_threaded,
        )


class ResponseTree(_TreeBase, collections.abc.Mapping):
    """
    A class used to represent a collection of CGroup directories created on the target system,
    abstracting the lower level complexities and allowing to user to interact with the CGroups.
    The structure of the tree mirrors the structure of the :class:`RequestTree` object tree used to create it, where
    each :class:`ResponseTree` object represents and abstracts the low-level CGroups its respective :class:`RequestTree` object defines.

    :param name: Name assigned to the :class:`ResponseTree` object, mirrors the name defined to its respective :class:`RequestTree` Object.
    :type name: str

    :param children: A dictionary that maps children names that this :class:`ResponseTree` object is a parent to
        and the respective :class:`ResponseTree` object the names represent.
    :type children: dict[str,:class:`ResponseTree`]

    :param low_levels: A dictionary that maps CGroup controller names to the suitable low level CGroup this :class:`ResponseTree` abstracts.
    :type low_levels: Dict[str, :class:`_CGroupBase`]

    :param user_low_levels: A dictionary that maps CGroup controller names to the suitable low level CGroup the
        :class:`RequestTree` object this class mirrors has specified. This is used within the context of a V1 user
        defined hierarchy in order to abstract the additional CGroups this class represents when trying to ensure V2 semantic
        equivalence. Done purely for cosmetic reasons.
    :type user_low_levels: Dict[str, :class:`_CGroupBase`]

    :param is_threaded: Boolean flag representing whether or not this ResponseTree object represents a single threaded V2 CGroup
        or a collection of pseudo-threaded V1 CGroups.
    :type is_threaded: bool
    """

    def __init__(
        self,
        name: str,
        children: Dict[str, _TreeBase],
        low_levels: Dict[str, _CGroupBase],
        user_low_levels: Dict[str, _CGroupBase],
        is_threaded: bool,
    ):
        self.children = children
        self.low_levels = low_levels
        self.user_low_levels = user_low_levels
        super().__init__(name=name, is_threaded=is_threaded)

    @property
    def _node_information(self):
        # Returns a formatted string, displaying the enabled user-defined controllers and their paths
        # (alongside the type of CGroup the controller resides in).
        return ", ".join(
            "{controller}@{path} [{cgroup_type}]".format(
                controller=controller,
                path=low_level.group_path,
                cgroup_type=self.group_type,
            )
            for controller, low_level in self.user_low_levels.items()
        )

    @property
    def _children_list(self):
        # Children Objects are the values in our self.children dict.
        return list(self.children.values())

    @property
    def _all_nodes(self):
        return list(
            itertools.chain(
                self.low_levels.values(),
                itertools.chain.from_iterable(
                    map(lambda child: child._all_nodes, self.children.values()),
                ),
            )
        )

    def add_process(self, pid: int):
        """
        Adds the process associated with ``pid`` to the low level CGroups this :class:`ResponseTree` object represents.

        :param pid: the PID of the process to be added to the low-level CGroups.
        :type pid: int

        :raises TargetStableError: Occurs in the case where this object is a parent to non-threaded children.
            Ensures V2 hierarchy compatibility.
        """

        if self.is_threaded_domain or self.is_threaded or not self.children:
            for low_level in self.low_levels.values():
                low_level._add_process(pid=pid)
        else:
            raise TargetStableError(
                "Cannot add Process ID: {pid} to {name}. The ResponseTree object is a parent to a non-threaded ResponseTree.".format(
                    pid=pid, name=self.name
                )
            )

    def add_thread(self, tid: int):
        """
        Adds the thread associated with the ``tid`` to the low level CGroups this :class:`ResponseTree` object represents.

        :param tid: the TID of the thread to be added to the low-level CGroups.
        :type tid: int

        :raises TargetStableError: Occurs in the case where this object is not threaded.
            Ensures V2 hierarchy compatibility.
        """

        if self.is_threaded:
            for lower_level in set(self.low_levels.values()):
                lower_level._add_thread(tid, self.threaded_domain)
        else:
            raise TargetStableError(
                "Cannot add Thread ID: {tid} to {name}. The ResponseTree object is not threaded.".format(
                    tid=tid, name=self.name
                )
            )

    def __getitem__(self, child_name: str):
        return self.children[child_name]

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)
