ALTER TABLE jobs
    DROP CONSTRAINT jobs_run_oid_fkey,
    ADD CONSTRAINT jobs_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;

ALTER TABLE targets
    DROP CONSTRAINT targets_run_oid_fkey,
    ADD CONSTRAINT targets_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;

ALTER TABLE events
    DROP CONSTRAINT events_run_oid_fkey,
    ADD CONSTRAINT events_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;

ALTER TABLE resource_getters
    DROP CONSTRAINT resource_getters_run_oid_fkey,
    ADD CONSTRAINT resource_getters_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;

ALTER TABLE augmentations
    DROP CONSTRAINT augmentations_run_oid_fkey,
    ADD CONSTRAINT augmentations_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;

ALTER TABLE jobs_augs
    DROP CONSTRAINT jobs_augs_job_oid_fkey,
    DROP CONSTRAINT jobs_augs_augmentation_oid_fkey,
    ADD CONSTRAINT jobs_augs_job_oid_fkey
        FOREIGN KEY (job_oid)
        REFERENCES Jobs(oid)
        ON DELETE CASCADE,
    ADD CONSTRAINT jobs_augs_augmentation_oid_fkey
        FOREIGN KEY (augmentation_oid)
        REFERENCES Augmentations(oid)
        ON DELETE CASCADE
;

ALTER TABLE metrics
    DROP CONSTRAINT metrics_run_oid_fkey,
    ADD CONSTRAINT metrics_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;

ALTER TABLE artifacts
    DROP CONSTRAINT artifacts_run_oid_fkey,
    ADD CONSTRAINT artifacts_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;

CREATE RULE del_lo AS
    ON DELETE TO Artifacts
    DO DELETE FROM LargeObjects
        WHERE LargeObjects.oid = old.large_object_uuid
;

ALTER TABLE classifiers
    DROP CONSTRAINT classifiers_artifact_oid_fkey,
    DROP CONSTRAINT classifiers_metric_oid_fkey,
    DROP CONSTRAINT classifiers_job_oid_fkey,
    DROP CONSTRAINT classifiers_run_oid_fkey,

    ADD CONSTRAINT classifiers_artifact_oid_fkey
        FOREIGN KEY (artifact_oid)
        REFERENCES artifacts(oid)
        ON DELETE CASCADE,

    ADD CONSTRAINT classifiers_metric_oid_fkey
        FOREIGN KEY (metric_oid)
        REFERENCES metrics(oid)
        ON DELETE CASCADE,

    ADD CONSTRAINT classifiers_job_oid_fkey
        FOREIGN KEY (job_oid)
        REFERENCES jobs(oid)
        ON DELETE CASCADE,

    ADD CONSTRAINT classifiers_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;

ALTER TABLE parameters
    DROP CONSTRAINT parameters_run_oid_fkey,
    ADD CONSTRAINT parameters_run_oid_fkey
        FOREIGN KEY (run_oid)
        REFERENCES runs(oid)
        ON DELETE CASCADE
;
