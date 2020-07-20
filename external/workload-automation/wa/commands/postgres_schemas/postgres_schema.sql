--!VERSION!1.6!ENDVERSION!
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "lo";

-- In future, it may be useful to implement rules on which Parameter oid fields can be none depeendent on the value in the type column;

DROP TABLE IF EXISTS DatabaseMeta;
DROP TABLE IF EXISTS Parameters;
DROP TABLE IF EXISTS Classifiers;
DROP TABLE IF EXISTS LargeObjects;
DROP TABLE IF EXISTS Artifacts;
DROP TABLE IF EXISTS Metrics;
DROP TABLE IF EXISTS Augmentations;
DROP TABLE IF EXISTS Jobs_Augs;
DROP TABLE IF EXISTS ResourceGetters;
DROP TABLE IF EXISTS Resource_Getters;
DROP TABLE IF EXISTS Events;
DROP TABLE IF EXISTS Targets;
DROP TABLE IF EXISTS Jobs;
DROP TABLE IF EXISTS Runs;

DROP TYPE IF EXISTS status_enum;
DROP TYPE IF EXISTS param_enum;

CREATE TYPE status_enum AS ENUM ('UNKNOWN(0)','NEW(1)','PENDING(2)','STARTED(3)','CONNECTED(4)', 'INITIALIZED(5)', 'RUNNING(6)', 'OK(7)', 'PARTIAL(8)', 'FAILED(9)', 'ABORTED(10)', 'SKIPPED(11)');

CREATE TYPE param_enum AS ENUM ('workload', 'resource_getter', 'augmentation', 'device', 'runtime', 'boot');

-- In future, it might be useful to create an ENUM type for the artifact kind, or simply a generic enum type;

CREATE TABLE DatabaseMeta (
    oid uuid NOT NULL,
    schema_major int,
    schema_minor int,
    PRIMARY KEY (oid)
);

CREATE TABLE Runs (
    oid uuid NOT NULL,
    event_summary text,
    basepath text,
    status status_enum,
    timestamp timestamp,
    run_name text,
    project text,
    project_stage text,
    retry_on_status status_enum[],
    max_retries int,
    bail_on_init_failure boolean,
    allow_phone_home boolean,
    run_uuid uuid,
    start_time timestamp,
    end_time timestamp,
    duration float,
    metadata jsonb,
    _pod_version int,
    _pod_serialization_version int,
    state jsonb,
    PRIMARY KEY (oid)
);

CREATE TABLE Jobs (
    oid uuid NOT NULL,
    run_oid uuid NOT NULL references Runs(oid) ON DELETE CASCADE,
    status status_enum,
    retry int,
    label text,
    job_id text,
    iterations int,
    workload_name text,
    metadata jsonb,
    _pod_version int,
    _pod_serialization_version int,
    PRIMARY KEY (oid)
);

CREATE TABLE Targets (
    oid uuid NOT NULL,
    run_oid uuid NOT NULL references Runs(oid) ON DELETE CASCADE,
    target text,
    modules text[],
    cpus text[],
    os text,
    os_version jsonb,
    hostid bigint,
    hostname text,
    abi text,
    is_rooted boolean,
    kernel_version text,
    kernel_release text,
    kernel_sha1 text,
    kernel_config text[],
    sched_features text[],
    page_size_kb int,
    screen_resolution int[],
    prop json,
    android_id text,
    _pod_version int,
    _pod_serialization_version int,
    system_id text,
    PRIMARY KEY (oid)
);

CREATE TABLE Events (
    oid uuid NOT NULL,
    run_oid uuid NOT NULL references Runs(oid) ON DELETE CASCADE,
    job_oid uuid references Jobs(oid),
    timestamp timestamp,
    message text,
    _pod_version int,
    _pod_serialization_version int,
    PRIMARY KEY (oid)
);

CREATE TABLE Resource_Getters (
    oid uuid NOT NULL,
    run_oid uuid NOT NULL references Runs(oid) ON DELETE CASCADE,
    name text,
    PRIMARY KEY (oid)
);

CREATE TABLE Augmentations (
    oid uuid NOT NULL,
    run_oid uuid NOT NULL references Runs(oid) ON DELETE CASCADE,
    name text,
    PRIMARY KEY (oid)
);

CREATE TABLE Jobs_Augs (
    oid uuid NOT NULL,
    job_oid uuid NOT NULL references Jobs(oid) ON DELETE CASCADE,
    augmentation_oid uuid NOT NULL references Augmentations(oid) ON DELETE CASCADE,
    PRIMARY KEY (oid)
);

CREATE TABLE Metrics (
    oid uuid NOT NULL,
    run_oid uuid NOT NULL references Runs(oid) ON DELETE CASCADE,
    job_oid uuid references Jobs(oid),
    name text,
    value double precision,
    units text,
    lower_is_better boolean,
    _pod_version int,
    _pod_serialization_version int,
    PRIMARY KEY (oid)
);

CREATE TABLE LargeObjects (
    oid uuid NOT NULL,
    lo_oid lo NOT NULL,
    PRIMARY KEY (oid)
);

-- Trigger that allows you to manage large objects from the LO table directly;
CREATE TRIGGER t_raster BEFORE UPDATE OR DELETE ON LargeObjects
    FOR EACH ROW EXECUTE PROCEDURE lo_manage(lo_oid);

CREATE TABLE Artifacts (
    oid uuid NOT NULL,
    run_oid uuid NOT NULL references Runs(oid) ON DELETE CASCADE,
    job_oid uuid references Jobs(oid),
    name text,
    large_object_uuid uuid NOT NULL references LargeObjects(oid),
    description text,
    kind text,
    _pod_version int,
    _pod_serialization_version int,
    is_dir boolean,
    PRIMARY KEY (oid)
);

CREATE RULE del_lo AS
    ON DELETE TO Artifacts
    DO DELETE FROM LargeObjects
        WHERE LargeObjects.oid = old.large_object_uuid
;

CREATE TABLE Classifiers (
    oid uuid NOT NULL,
    artifact_oid uuid references Artifacts(oid) ON DELETE CASCADE,
    metric_oid uuid references Metrics(oid) ON DELETE CASCADE,
    job_oid uuid references Jobs(oid) ON DELETE CASCADE,
    run_oid uuid references Runs(oid) ON DELETE CASCADE,
    key text,
    value text,
    PRIMARY KEY (oid)
);

CREATE TABLE Parameters (
    oid uuid NOT NULL,
    run_oid uuid NOT NULL references Runs(oid) ON DELETE CASCADE,
    job_oid uuid references Jobs(oid),
    augmentation_oid uuid references Augmentations(oid),
    resource_getter_oid uuid references Resource_Getters(oid),
    name text,
    value text,
    value_type text,
    type param_enum,
    PRIMARY KEY (oid)
);
