#    Copyright 2013-2018 ARM Limited
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

# pylint: disable=attribute-defined-outside-init

import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from contextlib import contextmanager

from wa import OutputProcessor, Parameter, OutputProcessorError
from wa.utils.serializer import json
from wa.utils.types import boolean


# IMPORTANT: when updating this schema, make sure to bump the version!
SCHEMA_VERSION = '0.0.2'
SCHEMA = [
    '''CREATE TABLE  runs (
        uuid text,
        start_time datetime,
        end_time datetime,
        duration integer
    )''',
    '''CREATE TABLE  workload_specs (
        id text,
        run_oid text,
        number_of_iterations integer,
        label text,
        workload_name text,
        boot_parameters text,
        runtime_parameters text,
        workload_parameters text
    )''',
    '''CREATE TABLE  metrics (
        spec_oid int,
        iteration integer,
        metric text,
        value text,
        units text,
        lower_is_better integer
    )''',
    '''CREATE VIEW results AS
       SELECT uuid as run_uuid, spec_id, label as workload, iteration, metric, value, units, lower_is_better
       FROM metrics AS m INNER JOIN (
            SELECT ws.OID as spec_oid, ws.id as spec_id, uuid, label
            FROM workload_specs AS ws INNER JOIN runs AS r ON ws.run_oid = r.OID
       ) AS wsr ON wsr.spec_oid = m.spec_oid
    ''',
    '''CREATE TABLE  __meta (
        schema_version text
    )''',
    '''INSERT INTO __meta VALUES ("{}")'''.format(SCHEMA_VERSION),
]


sqlite3.register_adapter(datetime, lambda x: x.isoformat())
sqlite3.register_adapter(timedelta, lambda x: x.total_seconds())
sqlite3.register_adapter(uuid.UUID, str)


class SqliteResultProcessor(OutputProcessor):

    name = 'sqlite'
    description = """
    Stores results in an sqlite database.

    This may be used to accumulate results of multiple runs in a single file.

    """
    parameters = [
        Parameter('database', default=None,
                  global_alias='sqlite_database',
                  description="""
                  Full path to the sqlite database to be used. If this is not
                  specified then a new database file will be created in the
                  output directory. This setting can be used to accumulate
                  results from multiple runs in a single database. If the
                  specified file does not exist, it will be created, however
                  the directory of the file must exist.

                  .. note:: The value must resolve to an absolute path,
                            relative paths are not allowed; however the
                            value may contain environment variables and/or
                            the home reference "~".
                  """),
        Parameter('overwrite', kind=boolean, default=False,
                  global_alias='sqlite_overwrite',
                  description="""
                  If ``True``, this will overwrite the database file
                  if it already exists. If ``False`` (the default) data
                  will be added to the existing file (provided schema
                  versions match -- otherwise an error will be raised).
                  """),

    ]

    def __init__(self, *args, **kwargs):
        super(SqliteResultProcessor, self).__init__(*args, **kwargs)
        self._last_spec = None
        self._run_oid = None
        self._spec_oid = None
        self._run_initialized = False

    def export_job_output(self, job_output, target_info, run_output):  # pylint: disable=unused-argument
        if not self._run_initialized:
            self._init_run(run_output)

        if self._last_spec != job_output.spec:
            self._update_spec(job_output.spec)

        metrics = [(self._spec_oid, job_output.iteration, m.name, str(m.value), m.units, int(m.lower_is_better))
                   for m in job_output.metrics]
        if metrics:
            with self._open_connection() as conn:
                conn.executemany('INSERT INTO metrics VALUES (?,?,?,?,?,?)', metrics)

    def export_run_output(self, run_output, target_info):  # pylint: disable=unused-argument
        if not self._run_initialized:
            self._init_run(run_output)

        metrics = [(self._spec_oid, run_output.iteration, m.name, str(m.value), m.units, int(m.lower_is_better))
                   for m in run_output.metrics]
        if metrics:
            with self._open_connection() as conn:
                conn.executemany('INSERT INTO metrics VALUES (?,?,?,?,?,?)', metrics)

        info = run_output.info
        with self._open_connection() as conn:
            conn.execute('''UPDATE runs SET start_time=?, end_time=?, duration=?
                            WHERE OID=?''', (info.start_time, info.end_time, info.duration, self._run_oid))

    def _init_run(self, run_output):
        if not self.database:  # pylint: disable=access-member-before-definition
            self.database = os.path.join(run_output.basepath, 'results.sqlite')
        self.database = os.path.expandvars(os.path.expanduser(self.database))

        if not os.path.exists(self.database):
            self._init_db()
        elif self.overwrite:  # pylint: disable=no-member
            os.remove(self.database)
            self._init_db()
        else:
            self._validate_schema_version()
        self._update_run(run_output.info.uuid)

        # if the database file happens to be in the output directory, add it as an
        # artifiact; if it isn't, then RunOutput doesn't need to keep track of it.
        if not os.path.relpath(self.database, run_output.basepath).startswith('..'):
            run_output.add_artifact('sqlitedb', self.database, kind='export')

        self._run_initialized = True

    def _init_db(self):
        with self._open_connection() as conn:
            for command in SCHEMA:
                conn.execute(command)

    def _validate_schema_version(self):
        with self._open_connection() as conn:
            try:
                c = conn.execute('SELECT schema_version FROM __meta')
                found_version = c.fetchone()[0]
            except sqlite3.OperationalError:
                message = '{} does not appear to be a valid WA results database.'.format(self.database)
                raise OutputProcessorError(message)
            if found_version != SCHEMA_VERSION:
                message = 'Schema version in {} ({}) does not match current version ({}).'
                raise OutputProcessorError(message.format(self.database, found_version, SCHEMA_VERSION))

    def _update_run(self, run_uuid):
        with self._open_connection() as conn:
            conn.execute('INSERT INTO runs (uuid) VALUES (?)', (run_uuid,))
            conn.commit()
            c = conn.execute('SELECT OID FROM runs WHERE uuid=?', (run_uuid,))
            self._run_oid = c.fetchone()[0]

    def _update_spec(self, spec):
        self._last_spec = spec
        spec_tuple = (spec.id, self._run_oid, spec.iterations, spec.label, spec.workload_name,
                      json.dumps(spec.boot_parameters.to_pod()),
                      json.dumps(spec.runtime_parameters.to_pod()),
                      json.dumps(spec.workload_parameters.to_pod()))
        with self._open_connection() as conn:
            conn.execute('INSERT INTO workload_specs VALUES (?,?,?,?,?,?,?,?)', spec_tuple)
            conn.commit()
            c = conn.execute('SELECT OID FROM workload_specs WHERE run_oid=? AND id=?', (self._run_oid, spec.id))
            self._spec_oid = c.fetchone()[0]

    @contextmanager
    def _open_connection(self):
        conn = sqlite3.connect(self.database)
        try:
            yield conn
        finally:
            conn.commit()
