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
import re
import string
import tarfile

try:
    import pymongo
    from bson.objectid import ObjectId
    from gridfs import GridFS
except ImportError:
    pymongo = None

from wlauto import ResultProcessor, Parameter, Artifact
from wlauto.exceptions import ResultProcessorError
from wlauto.utils.misc import as_relative


__bad_chars = '$.'
KEY_TRANS_TABLE = string.maketrans(__bad_chars, '_' * len(__bad_chars))
BUNDLE_NAME = 'files.tar.gz'


class MongodbUploader(ResultProcessor):

    name = 'mongodb'
    description = """
    Uploads run results to a MongoDB instance.

    MongoDB is a popular document-based data store (NoSQL database).

    """

    parameters = [
        Parameter('uri', kind=str, default=None,
                  description="""Connection URI. If specified, this will be used for connecting
                                 to the backend, and host/port parameters will be ignored."""),
        Parameter('host', kind=str, default='localhost', mandatory=True,
                  description='IP address/name of the machinge hosting the MongoDB server.'),
        Parameter('port', kind=int, default=27017, mandatory=True,
                  description='Port on which the MongoDB server is listening.'),
        Parameter('db', kind=str, default='wa', mandatory=True,
                  description='Database on the server used to store WA results.'),
        Parameter('extra_params', kind=dict, default={},
                  description='''Additional connection parameters may be specfied using this (see
                                 pymongo documentation.'''),
        Parameter('authentication', kind=dict, default={},
                  description='''If specified, this will be passed to db.authenticate() upon connection;
                                 please pymongo documentaion authentication examples for detail.'''),
    ]

    def initialize(self, context):
        if pymongo is None:
            raise ResultProcessorError('mongodb result processor requres pymongo package to be installed.')
        try:
            self.client = pymongo.MongoClient(self.host, self.port, **self.extra_params)
        except pymongo.errors.PyMongoError, e:
            raise ResultProcessorError('Error connecting to mongod: {}'.fromat(e))
        self.dbc = self.client[self.db]
        self.fs = GridFS(self.dbc)
        if self.authentication:
            if not self.dbc.authenticate(**self.authentication):
                raise ResultProcessorError('Authentication to database {} failed.'.format(self.db))

        self.run_result_dbid = ObjectId()
        run_doc = context.run_info.to_dict()

        wa_adapter = run_doc['device']
        devprops = dict((k.translate(KEY_TRANS_TABLE), v)
                        for k, v in run_doc['device_properties'].iteritems())
        run_doc['device'] = devprops
        run_doc['device']['wa_adapter'] = wa_adapter
        del run_doc['device_properties']

        run_doc['output_directory'] = os.path.abspath(context.output_directory)
        run_doc['artifacts'] = []
        run_doc['workloads'] = context.config.to_dict()['workload_specs']
        for workload in run_doc['workloads']:
            workload['name'] = workload['workload_name']
            del workload['workload_name']
            workload['results'] = []
        self.run_dbid = self.dbc.runs.insert(run_doc)

        prefix = context.run_info.project if context.run_info.project else '[NOPROJECT]'
        run_part = context.run_info.run_name or context.run_info.uuid.hex
        self.gridfs_dir = os.path.join(prefix, run_part)
        i = 0
        while self.gridfs_directory_exists(self.gridfs_dir):
            if self.gridfs_dir.endswith('-{}'.format(i)):
                self.gridfs_dir = self.gridfs_dir[:-2]
            i += 1
            self.gridfs_dir += '-{}'.format(i)

        # Keep track of all generated artefacts, so that we know what to
        # include in the tarball. The tarball will contains raw artificats
        # (other kinds would have been uploaded directly or do not contain
        # new data) and all files in the results dir that have not been marked
        # as artificats.
        self.artifacts = []

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
        iteration_artefacts = [self.upload_artifact(context, a) for a in context.iteration_artifacts]
        r['artifacts'] = [e for e in iteration_artefacts if e is not None]
        self.dbc.runs.update({'_id': self.run_dbid, 'workloads.id': context.spec.id},
                             {'$push': {'workloads.$.results': r}})

    def export_run_result(self, result, context):
        run_artifacts = [self.upload_artifact(context, a) for a in context.run_artifacts]
        self.logger.debug('Generating results bundle...')
        bundle = self.generate_bundle(context)
        if bundle:
            run_artifacts.append(self.upload_artifact(context, bundle))
        else:
            self.logger.debug('No untracked files found.')
        run_stats = {
            'status': result.status,
            'events': [e.to_dict() for e in result.events],
            'end_time': context.run_info.end_time,
            'duration': context.run_info.duration.total_seconds(),
            'artifacts': [e for e in run_artifacts if e is not None],
        }
        self.dbc.runs.update({'_id': self.run_dbid}, {'$set': run_stats})

    def finalize(self, context):
        self.client.close()

    def validate(self):
        if self.uri:
            has_warned = False
            if self.host != self.parameters['host'].default:
                self.logger.warning('both uri and host specified; host will be ignored')
                has_warned = True
            if self.port != self.parameters['port'].default:
                self.logger.warning('both uri and port specified; port will be ignored')
                has_warned = True
            if has_warned:
                self.logger.warning('To supress this warning, please remove either uri or '
                                    'host/port from your config.')

    def upload_artifact(self, context, artifact):
        artifact_path = os.path.join(context.output_directory, artifact.path)
        self.artifacts.append((artifact_path, artifact))
        if not os.path.exists(artifact_path):
            self.logger.debug('Artifact {} has not been generated'.format(artifact_path))
            return
        elif artifact.kind in ['raw', 'export']:
            self.logger.debug('Ignoring {} artifact {}'.format(artifact.kind, artifact_path))
            return
        else:
            self.logger.debug('Uploading artifact {}'.format(artifact_path))
            entry = artifact.to_dict()
            path = entry['path']
            del entry['path']
            del entry['name']
            del entry['level']
            del entry['mandatory']

            if context.workload is None:
                entry['filename'] = os.path.join(self.gridfs_dir, as_relative(path))
            else:
                entry['filename'] = os.path.join(self.gridfs_dir,
                                                 '{}-{}-{}'.format(context.spec.id,
                                                                   context.spec.label,
                                                                   context.current_iteration),
                                                 as_relative(path))
            with open(artifact_path, 'rb') as fh:
                fsid = self.fs.put(fh, **entry)
                entry['gridfs_id'] = fsid

            return entry

    def gridfs_directory_exists(self, path):
        regex = re.compile('^{}'.format(path))
        return self.fs.exists({'filename': regex})

    def generate_bundle(self, context):  # pylint: disable=R0914
        """
        The bundle will contain files generated during the run that have not
        already been processed. This includes all files for which there isn't an
        explicit artifact as well as "raw" artifacts that aren't uploaded individually.
        Basically, this ensures that everything that is not explicilty marked as an
        "export" (which means it's guarnteed not to contain information not accessible
        from other artifacts/scores) is avialable in the DB. The bundle is compressed,
        so it shouldn't take up too much space, however it also means that it's not
        easy to query for or get individual file (a trade off between space and convinience).

        """
        to_upload = []
        artpaths = []
        outdir = context.output_directory
        for artpath, artifact in self.artifacts:
            artpaths.append(os.path.relpath(artpath, outdir))
            if artifact.kind == 'raw':
                to_upload.append((artpath, os.path.relpath(artpath, outdir)))
        for root, _, files in os.walk(outdir):
            for f in files:
                path = os.path.relpath(os.path.join(root, f), outdir)
                if path not in artpaths:
                    to_upload.append((os.path.join(outdir, path), path))

        if not to_upload:
            # Nothing unexpected/unprocessed has been generated during the run.
            return None
        else:
            archive_path = os.path.join(outdir, BUNDLE_NAME)
            with tarfile.open(archive_path, 'w:gz') as tf:
                for fpath, arcpath in to_upload:
                    tf.add(fpath, arcpath)
            return Artifact('mongo_bundle', BUNDLE_NAME, 'data',
                            description='bundle to be uploaded to mongodb.')
