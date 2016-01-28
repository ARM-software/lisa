#    Copyright 2014-2015 ARM Limited
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


#pylint: disable=E1101,W0201
import os
from base64 import b64encode

from wlauto import ResultProcessor, Parameter
from wlauto.utils.serializer import json
from wlauto.utils.misc import istextfile
from wlauto.utils.types import list_of_strings
from wlauto.exceptions import ResultProcessorError


class JsonReportProcessor(ResultProcessor):

    name = 'json'
    description = """
    Produces a JSON file with WA config, results ect.


    This includes embedded artifacts either as text or base64

    """

    parameters = [
        Parameter("ignore_artifact_types", kind=list_of_strings,
                  default=['export', 'raw'],
                  description="""A list of which artifact types to be ignored,
                                 and thus not embedded in the JSON""")
    ]
    final = {}

    def initialize(self, context):
        self.final = context.run_info.to_dict()
        del self.final['workload_specs']

        wa_adapter = self.final['device']
        self.final['device'] = {}
        self.final['device']['props'] = self.final['device_properties']
        self.final['device']['wa_adapter'] = wa_adapter
        del self.final['device_properties']

        self.final['output_directory'] = os.path.abspath(context.output_directory)
        self.final['artifacts'] = []
        self.final['workloads'] = context.config.to_dict()['workload_specs']
        for workload in self.final['workloads']:
            workload['name'] = workload['workload_name']
            del workload['workload_name']
            workload['results'] = []

    def export_iteration_result(self, result, context):
        r = {}
        r['iteration'] = context.current_iteration
        r['status'] = result.status
        r['events'] = [e.to_dict() for e in result.events]
        r['metrics'] = []
        for m in result.metrics:
            md = m.to_dict()
            md['is_summary'] = m.name in context.workload.summary_metrics
            r['metrics'].append(md)
        iteration_artefacts = [self.embed_artifact(context, a) for a in context.iteration_artifacts]
        r['artifacts'] = [e for e in iteration_artefacts if e is not None]
        for workload in self.final['workloads']:
            if workload['id'] == context.spec.id:
                workload.update(r)
                break
        else:
            raise ResultProcessorError("No workload spec with matching id found")

    def export_run_result(self, result, context):
        run_artifacts = [self.embed_artifact(context, a) for a in context.run_artifacts]
        self.logger.debug('Generating results bundle...')
        run_stats = {
            'status': result.status,
            'events': [e.to_dict() for e in result.events],
            'end_time': context.run_info.end_time,
            'duration': context.run_info.duration.total_seconds(),
            'artifacts': [e for e in run_artifacts if e is not None],
        }
        self.final.update(run_stats)
        json_path = os.path.join(os.path.abspath(context.output_directory), "run.json")
        with open(json_path, 'w') as json_file:
            json.dump(self.final, json_file)

    def embed_artifact(self, context, artifact):
        artifact_path = os.path.join(context.output_directory, artifact.path)

        if not os.path.exists(artifact_path):
            self.logger.debug('Artifact {} has not been generated'.format(artifact_path))
            return
        elif artifact.kind in self.ignore_artifact_types:
            self.logger.debug('Ignoring {} artifact {}'.format(artifact.kind, artifact_path))
            return
        else:
            self.logger.debug('Uploading artifact {}'.format(artifact_path))
            entry = artifact.to_dict()
            path = os.path.join(os.path.abspath(context.output_directory), entry['path'])
            if istextfile(open(path)):
                entry['encoding'] = "text"
                entry['content'] = open(path).read()
            else:
                entry['encoding'] = "base64"
                entry['content'] = b64encode(open(path).read())

            del entry['path']
            del entry['level']
            del entry['mandatory']
            return entry
