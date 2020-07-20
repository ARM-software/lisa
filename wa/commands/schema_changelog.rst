# 1
## 1.0
- First version
## 1.1
- LargeObjects table added as a substitute for the previous plan to
  use the filesystem and a path reference to store artifacts. This
  was done following an extended discussion and tests that verified
  that the savings in processing power were not enough to warrant
  the creation of a dedicated server or file handler.
## 1.2
- Rename the `resourcegetters` table to `resource_getters` for consistency.
- Add Job and Run level classifiers.
- Add missing android specific properties to targets.
- Add new POD meta data to relevant tables. 
- Correct job column name from `retires` to `retry`.
- Add missing run information.
## 1.3
- Add missing "system_id" field from TargetInfo.
- Enable support for uploading Artifact that represent directories.
## 1.4
- Add "modules" field to TargetInfo to list the modules loaded by the target
  during the run.
## 1.5
- Change the type of the "hostid" in TargetInfo from Int to Bigint.
## 1.6
- Add cascading deletes to most tables to allow easy deletion of a run
  and its associated data
- Add rule to delete associated large object on deletion of artifact