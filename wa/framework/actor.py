import uuid
import logging

from wa.framework import pluginloader
from wa.framework.plugin import Plugin


class JobActor(Plugin):

    kind = 'job_actor'

    def initialize(self, context):
        pass

    def run(self):
        pass

    def restart(self):
        pass

    def complete(self):
        pass

    def finalize(self):
        pass


class NullJobActor(JobActor):

    name = 'null-job-actor'

