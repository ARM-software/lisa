ALTER TABLE resourcegetters RENAME TO resource_getters;

ALTER TABLE classifiers ADD COLUMN job_oid uuid references Jobs(oid);
ALTER TABLE classifiers ADD COLUMN run_oid uuid references Runs(oid);

ALTER TABLE targets ADD COLUMN page_size_kb int;
ALTER TABLE targets ADD COLUMN screen_resolution int[];
ALTER TABLE targets ADD COLUMN prop text;
ALTER TABLE targets ADD COLUMN android_id text;
ALTER TABLE targets ADD COLUMN _pod_version int;
ALTER TABLE targets ADD COLUMN _pod_serialization_version int;

ALTER TABLE jobs RENAME COLUMN retries TO retry;
ALTER TABLE jobs ADD COLUMN _pod_version int;
ALTER TABLE jobs ADD COLUMN _pod_serialization_version int;

ALTER TABLE runs ADD COLUMN project_stage text;
ALTER TABLE runs ADD COLUMN state jsonb;
ALTER TABLE runs ADD COLUMN duration float;
ALTER TABLE runs ADD COLUMN _pod_version int;
ALTER TABLE runs ADD COLUMN _pod_serialization_version int;

ALTER TABLE artifacts ADD COLUMN _pod_version int;
ALTER TABLE artifacts ADD COLUMN _pod_serialization_version int;

ALTER TABLE events ADD COLUMN _pod_version int;
ALTER TABLE events ADD COLUMN _pod_serialization_version int;

ALTER TABLE metrics ADD COLUMN _pod_version int;
ALTER TABLE metrics ADD COLUMN _pod_serialization_version int;
