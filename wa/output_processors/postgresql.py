#    Copyright 2018 ARM Limited
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

import os
import uuid
import collections
import tarfile

try:
    import psycopg2
    from psycopg2 import (connect, extras)
    from psycopg2 import Error as Psycopg2Error
except ImportError as e:
    psycopg2 = None
    import_error_msg = e.args[0] if e.args else str(e)

from devlib.target import KernelVersion, KernelConfig

from wa import OutputProcessor, Parameter, OutputProcessorError
from wa.framework.target.info import CpuInfo
from wa.utils.postgres import (POSTGRES_SCHEMA_DIR, cast_level, cast_vanilla,
                               adapt_vanilla, return_as_is, adapt_level,
                               ListOfLevel, adapt_ListOfX, create_iterable_adapter,
                               get_schema_versions)
from wa.utils.serializer import json
from wa.utils.types import level


class PostgresqlResultProcessor(OutputProcessor):

    name = 'postgres'
    description = """
    Stores results in a Postgresql database.

    The structure of this database can easily be understood by examining
    the postgres_schema.sql file (the schema used to generate it):
    {}
    """.format(os.path.join(POSTGRES_SCHEMA_DIR, 'postgres_schema.sql'))

    parameters = [
        Parameter('username', default='postgres',
                  description="""
                  This is the username that will be used to connect to the
                  Postgresql database. Note that depending on whether the user
                  has privileges to modify the database (normally only possible
                  on localhost), the user may only be able to append entries.
                  """),
        Parameter('password', default=None,
                  description="""
                  The password to be used to connect to the specified database
                  with the specified username.
                  """),
        Parameter('dbname', default='wa',
                  description="""
                  Name of the database that will be created or added to. Note,
                  to override this, you can specify a value in your user
                  wa configuration file.
                  """),
        Parameter('host', kind=str, default='localhost',
                  description="""
                  The host where the Postgresql server is running. The default
                  is localhost (i.e. the machine that wa is running on).
                  This is useful for complex systems where multiple machines
                  may be executing workloads and uploading their results to
                  a remote, centralised database.
                  """),
        Parameter('port', kind=str, default='5432',
                  description="""
                  The port the Postgresql server is running on, on the host.
                  The default is Postgresql's default, so do not change this
                  unless you have modified the default port for Postgresql.
                  """),
    ]

    # Commands
    sql_command = {
        "create_run": "INSERT INTO Runs (oid, event_summary, basepath, status, timestamp, run_name, project, project_stage, retry_on_status, max_retries, bail_on_init_failure, allow_phone_home, run_uuid, start_time, metadata, state, _pod_version, _pod_serialization_version) "
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        "update_run": "UPDATE Runs SET event_summary=%s, status=%s, timestamp=%s, end_time=%s, duration=%s, state=%s WHERE oid=%s;",
        "create_job": "INSERT INTO Jobs (oid, run_oid, status, retry, label, job_id, iterations, workload_name, metadata, _pod_version, _pod_serialization_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
        "create_target": "INSERT INTO Targets (oid, run_oid, target, modules, cpus, os, os_version, hostid, hostname, abi, is_rooted, kernel_version, kernel_release, kernel_sha1, kernel_config, sched_features, page_size_kb, system_id, screen_resolution, prop, android_id, _pod_version, _pod_serialization_version) "
                         "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        "create_event": "INSERT INTO Events (oid, run_oid, job_oid, timestamp, message, _pod_version, _pod_serialization_version) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        "create_artifact": "INSERT INTO Artifacts (oid, run_oid, job_oid, name, large_object_uuid, description, kind, is_dir, _pod_version, _pod_serialization_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        "create_metric": "INSERT INTO Metrics (oid, run_oid, job_oid, name, value, units, lower_is_better, _pod_version, _pod_serialization_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        "create_augmentation": "INSERT INTO Augmentations (oid, run_oid, name) VALUES (%s, %s, %s)",
        "create_classifier": "INSERT INTO Classifiers (oid, artifact_oid, metric_oid, job_oid, run_oid, key, value) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        "create_parameter": "INSERT INTO Parameters (oid, run_oid, job_oid, augmentation_oid, resource_getter_oid, name, value, value_type, type) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        "create_resource_getter": "INSERT INTO Resource_Getters (oid, run_oid, name) VALUES (%s, %s, %s)",
        "create_job_aug": "INSERT INTO Jobs_Augs (oid, job_oid, augmentation_oid) VALUES (%s, %s, %s)",
        "create_large_object": "INSERT INTO LargeObjects (oid, lo_oid) VALUES (%s, %s)"
    }

    # Lists to track which run-related items have already been added
    metrics_already_added = []
    # Dicts needed so that jobs can look up ids
    artifacts_already_added = {}
    augmentations_already_added = {}

    # Status bits (flags)
    first_job_run = True

    def __init__(self, *args, **kwargs):
        super(PostgresqlResultProcessor, self).__init__(*args, **kwargs)
        self.conn = None
        self.cursor = None
        self.run_uuid = None
        self.target_uuid = None

    def initialize(self, context):

        if not psycopg2:
            raise ImportError(
                'The psycopg2 module is required for the '
                + 'Postgresql Output Processor: {}'.format(import_error_msg))
        # N.B. Typecasters are for postgres->python and adapters the opposite
        self.connect_to_database()

        # Register the adapters and typecasters for enum types
        self.cursor.execute("SELECT NULL::status_enum")
        status_oid = self.cursor.description[0][1]
        self.cursor.execute("SELECT NULL::param_enum")
        param_oid = self.cursor.description[0][1]
        LEVEL = psycopg2.extensions.new_type(
            (status_oid,), "LEVEL", cast_level)
        psycopg2.extensions.register_type(LEVEL)
        PARAM = psycopg2.extensions.new_type(
            (param_oid,), "PARAM", cast_vanilla)
        psycopg2.extensions.register_type(PARAM)
        psycopg2.extensions.register_adapter(level, return_as_is(adapt_level))
        psycopg2.extensions.register_adapter(
            ListOfLevel, adapt_ListOfX(adapt_level))
        psycopg2.extensions.register_adapter(KernelVersion, adapt_vanilla)
        psycopg2.extensions.register_adapter(
            CpuInfo, adapt_vanilla)
        psycopg2.extensions.register_adapter(
            collections.OrderedDict, extras.Json)
        psycopg2.extensions.register_adapter(dict, extras.Json)
        psycopg2.extensions.register_adapter(
            KernelConfig, create_iterable_adapter(2, explicit_iterate=True))
        # Register ready-made UUID type adapter
        extras.register_uuid()

        # Insert a run_uuid which will be globally accessible during the run
        self.run_uuid = uuid.UUID(str(uuid.uuid4()))
        run_output = context.run_output
        retry_on_status = ListOfLevel(run_output.run_config.retry_on_status)
        self.cursor.execute(
            self.sql_command['create_run'],
            (
                self.run_uuid,
                run_output.event_summary,
                run_output.basepath,
                run_output.status,
                run_output.state.timestamp,
                run_output.info.run_name,
                run_output.info.project,
                run_output.info.project_stage,
                retry_on_status,
                run_output.run_config.max_retries,
                run_output.run_config.bail_on_init_failure,
                run_output.run_config.allow_phone_home,
                run_output.info.uuid,
                run_output.info.start_time,
                run_output.metadata,
                json.dumps(run_output.state.to_pod()),
                run_output.result._pod_version,  # pylint: disable=protected-access
                run_output.result._pod_serialization_version,  # pylint: disable=protected-access
            )
        )
        self.target_uuid = uuid.uuid4()
        target_info = context.target_info
        target_pod = target_info.to_pod()
        self.cursor.execute(
            self.sql_command['create_target'],
            (
                self.target_uuid,
                self.run_uuid,
                target_pod['target'],
                target_pod['modules'],
                target_pod['cpus'],
                target_pod['os'],
                target_pod['os_version'],
                target_pod['hostid'],
                target_pod['hostname'],
                target_pod['abi'],
                target_pod['is_rooted'],
                # Important caveat: kernel_version is the name of the column in the Targets table
                # However, this refers to kernel_version.version, not to kernel_version as a whole
                target_pod['kernel_version'],
                target_pod['kernel_release'],
                target_info.kernel_version.sha1,
                target_info.kernel_config,
                target_pod['sched_features'],
                target_pod['page_size_kb'],
                target_pod['system_id'],
                # Android Specific
                list(target_pod.get('screen_resolution', [])),
                target_pod.get('prop'),
                target_pod.get('android_id'),
                target_pod.get('_pod_version'),
                target_pod.get('_pod_serialization_version'),
            )
        )

        # Commit cursor commands
        self.conn.commit()

    def export_job_output(self, job_output, target_info, run_output):   # pylint: disable=too-many-branches, too-many-statements, too-many-locals, unused-argument
        ''' Run once for each job to upload information that is
            updated on a job by job basis.
        '''
        # Ensure we're still connected to the database.
        self.connect_to_database()
        job_uuid = uuid.uuid4()
        # Create a new job
        self.cursor.execute(
            self.sql_command['create_job'],
            (
                job_uuid,
                self.run_uuid,
                job_output.status,
                job_output.retry,
                job_output.label,
                job_output.id,
                job_output.iteration,
                job_output.spec.workload_name,
                job_output.metadata,
                job_output.spec._pod_version,  # pylint: disable=protected-access
                job_output.spec._pod_serialization_version,  # pylint: disable=protected-access
            )
        )

        for classifier in job_output.classifiers:
            classifier_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_classifier'],
                (
                    classifier_uuid,
                    None,
                    None,
                    job_uuid,
                    None,
                    classifier,
                    job_output.classifiers[classifier]
                )
            )
        # Update the run table and run-level parameters
        self.cursor.execute(
            self.sql_command['update_run'],
            (
                run_output.event_summary,
                run_output.status,
                run_output.state.timestamp,
                run_output.info.end_time,
                None,
                json.dumps(run_output.state.to_pod()),
                self.run_uuid))
        for classifier in run_output.classifiers:
            classifier_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_classifier'],
                (
                    classifier_uuid,
                    None,
                    None,
                    None,
                    None,
                    self.run_uuid,
                    classifier,
                    run_output.classifiers[classifier]
                )
            )
        self.sql_upload_artifacts(run_output, record_in_added=True)
        self.sql_upload_metrics(run_output, record_in_added=True)
        self.sql_upload_augmentations(run_output)
        self.sql_upload_resource_getters(run_output)
        self.sql_upload_events(job_output, job_uuid=job_uuid)
        self.sql_upload_artifacts(job_output, job_uuid=job_uuid)
        self.sql_upload_metrics(job_output, job_uuid=job_uuid)
        self.sql_upload_job_augmentations(job_output, job_uuid=job_uuid)
        self.sql_upload_parameters(
            "workload",
            job_output.spec.workload_parameters,
            job_uuid=job_uuid)
        self.sql_upload_parameters(
            "runtime",
            job_output.spec.runtime_parameters,
            job_uuid=job_uuid)
        self.conn.commit()

    def export_run_output(self, run_output, target_info):  # pylint: disable=unused-argument, too-many-locals
        ''' A final export of the RunOutput that updates existing parameters
            and uploads ones which are only generated after jobs have run.
        '''
        if self.cursor is None:  # Output processor did not initialise correctly.
            return
        # Ensure we're still connected to the database.
        self.connect_to_database()

        # Update the job statuses following completion of the run
        for job in run_output.jobs:
            job_id = job.id
            job_status = job.status
            self.cursor.execute(
                "UPDATE Jobs SET status=%s WHERE job_id=%s and run_oid=%s",
                (
                    job_status,
                    job_id,
                    self.run_uuid
                )
            )

        run_uuid = self.run_uuid
        # Update the run entry after jobs have completed
        run_info_pod = run_output.info.to_pod()
        run_state_pod = run_output.state.to_pod()
        sql_command_update_run = self.sql_command['update_run']
        self.cursor.execute(
            sql_command_update_run,
            (
                run_output.event_summary,
                run_output.status,
                run_info_pod['start_time'],
                run_info_pod['end_time'],
                run_info_pod['duration'],
                json.dumps(run_state_pod),
                run_uuid,
            )
        )
        self.sql_upload_events(run_output)
        self.sql_upload_artifacts(run_output, check_uniqueness=True)
        self.sql_upload_metrics(run_output, check_uniqueness=True)
        self.sql_upload_augmentations(run_output)
        self.conn.commit()

    # Upload functions for use with both jobs and runs

    def sql_upload_resource_getters(self, output_object):
        for resource_getter in output_object.run_config.resource_getters:
            resource_getter_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_resource_getter'],
                (
                    resource_getter_uuid,
                    self.run_uuid,
                    resource_getter,
                )
            )
            self.sql_upload_parameters(
                'resource_getter',
                output_object.run_config.resource_getters[resource_getter],
                owner_id=resource_getter_uuid,
            )

    def sql_upload_events(self, output_object, job_uuid=None):
        for event in output_object.events:
            event_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_event'],
                (
                    event_uuid,
                    self.run_uuid,
                    job_uuid,
                    event.timestamp,
                    event.message,
                    event._pod_version,  # pylint: disable=protected-access
                    event._pod_serialization_version,  # pylint: disable=protected-access
                )
            )

    def sql_upload_job_augmentations(self, output_object, job_uuid=None):
        ''' This is a table which links the uuids of augmentations to jobs.
        Note that the augmentations table is prepopulated, leading to the necessity
        of an augmentaitions_already_added dictionary, which gives us the corresponding
        uuids.
        Augmentations which are prefixed by ~ are toggled off and not part of the job,
        therefore not added.
        '''
        for augmentation in output_object.spec.augmentations:
            if augmentation.startswith('~'):
                continue
            augmentation_uuid = self.augmentations_already_added[augmentation]
            job_aug_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_job_aug'],
                (
                    job_aug_uuid,
                    job_uuid,
                    augmentation_uuid,
                )
            )

    def sql_upload_augmentations(self, output_object):
        for augmentation in output_object.augmentations:
            if augmentation.startswith('~') or augmentation in self.augmentations_already_added:
                continue
            augmentation_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_augmentation'],
                (
                    augmentation_uuid,
                    self.run_uuid,
                    augmentation,
                )
            )
            self.sql_upload_parameters(
                'augmentation',
                output_object.run_config.augmentations[augmentation],
                owner_id=augmentation_uuid,
            )
            self.augmentations_already_added[augmentation] = augmentation_uuid

    def sql_upload_metrics(self, output_object, record_in_added=False, check_uniqueness=False, job_uuid=None):
        for metric in output_object.metrics:
            if metric in self.metrics_already_added and check_uniqueness:
                continue
            metric_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_metric'],
                (
                    metric_uuid,
                    self.run_uuid,
                    job_uuid,
                    metric.name,
                    metric.value,
                    metric.units,
                    metric.lower_is_better,
                    metric._pod_version,  # pylint: disable=protected-access
                    metric._pod_serialization_version,  # pylint: disable=protected-access
                )
            )
            for classifier in metric.classifiers:
                classifier_uuid = uuid.uuid4()
                self.cursor.execute(
                    self.sql_command['create_classifier'],
                    (
                        classifier_uuid,
                        None,
                        metric_uuid,
                        None,
                        None,
                        classifier,
                        metric.classifiers[classifier],
                    )
                )
            if record_in_added:
                self.metrics_already_added.append(metric)

    def sql_upload_artifacts(self, output_object, record_in_added=False, check_uniqueness=False, job_uuid=None):
        ''' Uploads artifacts to the database.
        record_in_added will record the artifacts added in artifacts_aleady_added
        check_uniqueness will ensure artifacts in artifacts_already_added do not get added again
        '''
        for artifact in output_object.artifacts:
            if artifact in self.artifacts_already_added and check_uniqueness:
                self.logger.debug('Skipping uploading {} as already added'.format(artifact))
                continue

            if artifact in self.artifacts_already_added:
                self._sql_update_artifact(artifact, output_object)
            else:
                self._sql_create_artifact(artifact, output_object, record_in_added, job_uuid)

    def sql_upload_parameters(self, parameter_type, parameter_dict, owner_id=None, job_uuid=None):
        # Note, currently no augmentation parameters are workload specific, but in the future
        # this may change
        augmentation_id = None
        resource_getter_id = None

        if parameter_type not in ['workload', 'resource_getter', 'augmentation', 'runtime']:
            # boot parameters are not yet implemented
            # device parameters are redundant due to the targets table
            raise NotImplementedError("{} is not a valid parameter type.".format(parameter_type))

        if parameter_type == "resource_getter":
            resource_getter_id = owner_id
        elif parameter_type == "augmentation":
            augmentation_id = owner_id

        for parameter in parameter_dict:
            parameter_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_parameter'],
                (
                    parameter_uuid,
                    self.run_uuid,
                    job_uuid,
                    augmentation_id,
                    resource_getter_id,
                    parameter,
                    json.dumps(parameter_dict[parameter]),
                    str(type(parameter_dict[parameter])),
                    parameter_type,
                )
            )

    def connect_to_database(self):
        dsn = "dbname={} user={} password={} host={} port={}".format(
            self.dbname, self.username, self.password, self.host, self.port)
        try:
            self.conn = connect(dsn=dsn)
        except Psycopg2Error as e:
            raise OutputProcessorError(
                "Database error, if the database doesn't exist, "
                + "please use 'wa create database' to create the database: {}".format(e))
        self.cursor = self.conn.cursor()
        self.verify_schema_versions()

    def execute_sql_line_by_line(self, sql):
        cursor = self.conn.cursor()
        for line in sql.replace('\n', "").replace(";", ";\n").split("\n"):
            if line and not line.startswith('--'):
                cursor.execute(line)
        cursor.close()
        self.conn.commit()
        self.conn.reset()

    def verify_schema_versions(self):
        local_schema_version, db_schema_version = get_schema_versions(self.conn)
        if local_schema_version != db_schema_version:
            self.cursor.close()
            self.cursor = None
            self.conn.commit()
            self.conn.reset()
            msg = 'The current database schema is v{} however the local ' \
                  'schema version is v{}. Please update your database ' \
                  'with the create command'
            raise OutputProcessorError(msg.format(db_schema_version, local_schema_version))

    def _sql_write_file_lobject(self, source, lobject):
        with open(source) as lobj_file:
            lobj_data = lobj_file.read()
        if len(lobj_data) > 50000000:  # Notify if LO inserts larger than 50MB
            self.logger.debug("Inserting large object of size {}".format(len(lobj_data)))
        lobject.write(lobj_data)
        self.conn.commit()

    def _sql_write_dir_lobject(self, source, lobject):
        with tarfile.open(fileobj=lobject, mode='w|gz') as lobj_dir:
            lobj_dir.add(source, arcname='.')
        self.conn.commit()

    def _sql_update_artifact(self, artifact, output_object):
        self.logger.debug('Updating artifact: {}'.format(artifact))
        lobj = self.conn.lobject(oid=self.artifacts_already_added[artifact], mode='w')
        if artifact.is_dir:
            self._sql_write_dir_lobject(os.path.join(output_object.basepath, artifact.path), lobj)
        else:
            self._sql_write_file_lobject(os.path.join(output_object.basepath, artifact.path), lobj)

    def _sql_create_artifact(self, artifact, output_object, record_in_added=False, job_uuid=None):
        self.logger.debug('Uploading artifact: {}'.format(artifact))
        artifact_uuid = uuid.uuid4()
        lobj = self.conn.lobject()
        loid = lobj.oid
        large_object_uuid = uuid.uuid4()
        if artifact.is_dir:
            self._sql_write_dir_lobject(os.path.join(output_object.basepath, artifact.path), lobj)
        else:
            self._sql_write_file_lobject(os.path.join(output_object.basepath, artifact.path), lobj)

        self.cursor.execute(
            self.sql_command['create_large_object'],
            (
                large_object_uuid,
                loid,
            )
        )
        self.cursor.execute(
            self.sql_command['create_artifact'],
            (
                artifact_uuid,
                self.run_uuid,
                job_uuid,
                artifact.name,
                large_object_uuid,
                artifact.description,
                str(artifact.kind),
                artifact.is_dir,
                artifact._pod_version,  # pylint: disable=protected-access
                artifact._pod_serialization_version,  # pylint: disable=protected-access
            )
        )
        for classifier in artifact.classifiers:
            classifier_uuid = uuid.uuid4()
            self.cursor.execute(
                self.sql_command['create_classifier'],
                (
                    classifier_uuid,
                    artifact_uuid,
                    None,
                    None,
                    None,
                    classifier,
                    artifact.classifiers[classifier],
                )
            )
        if record_in_added:
            self.artifacts_already_added[artifact] = loid
