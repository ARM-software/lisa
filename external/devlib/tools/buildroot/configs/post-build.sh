#!/bin/sh

set -eux

# Enable root login on SSH
sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' "${TARGET_DIR}/etc/ssh/sshd_config"

# Increase the number of available channels so that devlib async code can
# exploit concurrency better.
sed -i 's/#MaxSessions.*/MaxSessions 100/' "${TARGET_DIR}/etc/ssh/sshd_config"
sed -i 's/#MaxStartups.*/MaxStartups 100/' "${TARGET_DIR}/etc/ssh/sshd_config"

# To test Android bindings of ChromeOsTarget
mkdir -p "${TARGET_DIR}/opt/google/containers/android"

