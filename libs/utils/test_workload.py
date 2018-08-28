# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

from serialize import YAMLSerializable

# from wa import Metric

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! FIXME !!! this is from workload-automation you bloody thief!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class Metric(object):
    """
    This is a single metric collected from executing a workload.

    :param name: the name of the metric. Uniquely identifies the metric
                 within the results.
    :param value: The numerical value of the metric for this execution of a
                  workload. This can be either an int or a float.
    :param units: Units for the collected value. Can be None if the value
                  has no units (e.g. it's a count or a standardised score).
    :param lower_is_better: Boolean flag indicating where lower values are
                            better than higher ones. Defaults to False.
    :param classifiers: A set of key-value pairs to further classify this
                        metric beyond current iteration (e.g. this can be used
                        to identify sub-tests).

    """

    __slots__ = ['name', 'value', 'units', 'lower_is_better', 'classifiers']

    @staticmethod
    def from_pod(pod):
        return Metric(**pod)

    def __init__(self, name, value, units=None, lower_is_better=False,
                 classifiers=None):
        self.name = name
        self.value = value
        self.units = units
        self.lower_is_better = lower_is_better
        self.classifiers = classifiers or {}

    def to_pod(self):
        return dict(
            name=self.name,
            value=self.value,
            units=self.units,
            lower_is_better=self.lower_is_better,
            classifiers=self.classifiers,
        )

    def __str__(self):
        result = '{}: {}'.format(self.name, self.value)
        if self.units:
            result += ' ' + self.units
        result += ' ({})'.format('-' if self.lower_is_better else '+')
        return result

    def __repr__(self):
        text = self.__str__()
        if self.classifiers:
            return '<{} {}>'.format(text, self.classifiers)
        else:
            return '<{}>'.format(text)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! FIXME !!! this is from workload-automation you bloody thief!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class ResultBundle(object):
    """
    Bundle for storing test results

    :param passed: Indicates whether the associated test passed.
      It will also be used as the truth-value of a ResultBundle.
    :type passed: boolean
    """
    def __init__(self, passed):
        self.passed = passed
        self.metrics = []

    def __nonzero__(self):
        return self.passed

    def add_metric(self, metric):
        """
        Lets you append several test :class:`Metric` to the bundle.

        :param metric: Metric to add to the bundle
        :type metric: Metric
        """
        self.metrics.append(metric)

class TestBundle(YAMLSerializable):
    """
    A LISA test bundle.

    :param res_dir: Directory from where the target execution artifacts reside.
        This will also be used to dump any artifact generated in the test code.
    :type res_dir: str

    **Design notes:**

    * :meth:`from_target` will collect whatever artifacts are required
      from a given target, and will then return a :class:`TestBundle`.
    * :meth:`from_path` will use whatever artifacts are available in a
      given directory (which should have been created by an earlier call
      to :meth:`from_target` and then :meth:`to_path`), and will then return
      a :class:`TestBundle`.
    * :attr:`verify_serialization` is there to ensure both above methods remain
      operationnal at all times.

    The point of a TestBundle is to bundle in a single object all of the
    required data to run some test assertion (hence the name). When inheriting
    from this class, you can define test methods that use this data, and return
    a :class:`ResultBundle`.

    **Implementation example**::

        class DummyTestBundle(TestBundle):

            def __init__(self, res_dir, shell_output):
                super(DummyTestBundle, self).__init__(res_dir)

                self.shell_output = shell_output

            @classmethod
            def _from_target(cls, te, res_dir):
                output = te.target.execute('echo $((21+21))').split()
                return cls(res_dir, output)

            def test_output(self):
                passed = False
                for line in self.shell_output:
                    if '42' in line:
                        passed = True
                        break

                return ResultBundle(passed)

    **Usage example**::

        # Creating a Bundle from a live target
        bundle = TestBundle.from_target(test_env, "/my/res/dir")
        # Running some test on the bundle
        res_bundle = bundle.test_foo()

        # Saving the bundle on the disk
        bundle.to_path(test_env, "/my/res/dir")

        # Reloading the bundle from the disk
        bundle = TestBundle.from_path("/my/res/dir")
        res_bundle = bundle.test_foo()
    """

    verify_serialization = True
    """
    When True, this enforces a serialization/deserialization step in :meth:`from_target`.
    Although it hinders performance (we end up creating two :class:`TestBundle`
    instances), it's very valuable to ensure :meth:`from_path` does not get broken
    for some particular class.
    """

    def __init__(self, res_dir):
        self.res_dir = res_dir

    @classmethod
    def _from_target(cls, te, res_dir):
        """
        Internals of the target factory method.
        """
        raise NotImplementedError()

    @classmethod
    def from_target(cls, te, res_dir=None, **kwargs):
        """
        Factory method to create a bundle using a live target

        This is mostly boiler-plate code around :meth:`_from_target`,
        which lets us introduce common functionalities for daughter classes.
        Unless you know what you are doing, you should not override this method,
        but the internal :meth:`_from_target` instead.
        """
        if not res_dir:
            res_dir = te.get_res_dir()

        # Logger stuff?

        bundle = cls._from_target(te, res_dir, **kwargs)

        # We've created the bundle from the target, and have all of
        # the information we need to execute the test code. However,
        # we enforce the use of the offline reloading path to ensure
        # it does not get broken.
        if cls.verify_serialization:
            bundle.to_path(res_dir)
            bundle = cls.from_path(res_dir)

        return bundle

    @classmethod
    def _filepath(cls, res_dir):
        return os.path.join(res_dir, "{}.yaml".format(cls.__name__))

    @classmethod
    def from_path(cls, res_dir):
        """
        See :meth:`YAMLSerializable.from_path`
        """
        bundle = YAMLSerializable.from_path(cls._filepath(res_dir))
        # We need to update the res_dir to the one we were given
        bundle.res_dir = res_dir

        return bundle

    def to_path(self, res_dir):
        """
        See :meth:`YAMLSerializable.to_path`
        """
        super(TestBundle, self).to_path(self._filepath(res_dir))

################################################################################
################################################################################

# class AndroidWorkload(LisaWorkload):

#     def _setup_wload(self):
#         self.target.set_auto_brightness(0)
#         self.target.set_brightness(0)

#         self.target.ensure_screen_is_on()
#         self.target.swipe_to_unlock()

#         self.target.set_auto_rotation(0)
#         self.target.set_rotation(1)

#     def _run_wload(self):
#         pass

#     def _teardown_wload(self):
#         self.target.set_auto_rotation(1)
#         self.target.set_auto_brightness(1)

#     def run(self, trace_tool):
#         if trace_tool == "ftrace":
#             pass
#         elif trace_tool == "systrace":
#             pass

#         self._setup_wload()

#         with self.te.record_ftrace():
#             self._run_wload()

#         self._teardown_wload()

# from target_script import TargetScript
# from devlib.target import AndroidTarget

# class GmapsWorkload(AndroidWorkload):

#     def _setup_wload(self):
#         super(GmapsWorkload, self)._setup_wload()

#         self.script = TargetScript(self.te, "gmaps_swiper.sh")

#         for i in range(self.swipe_count):
#             # Swipe right
#             self.script.input_swipe_pct(40, 50, 60, 60)
#             #AndroidTarget.input_swipe_pct(self.script, 40, 50, 60, 60)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe down
#             AndroidTarget.input_swipe_pct(self.script, 50, 60, 50, 40)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe left
#             AndroidTarget.input_swipe_pct(self.script, 60, 50, 40, 50)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe up
#             AndroidTarget.input_swipe_pct(self.script, 50, 40, 50, 60)
#             AndroidTarget.sleep(self.script, 1)

#         # Push script to the target
#         self.script.push()

#     def _run_wload(self):
#         self.script.run()

#     def run(self, swipe_count=10):
#         self.swipe_count = swipe_count

#         super(GmapsWorkload, self).run("ftrace")
