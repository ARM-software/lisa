#!/bin/sh

# Control groups mount point
CGMOUNT=${CGMOUNT:-/sys/fs/cgroup}
# The control group we want to run into
CGP=${1}
# The command to run
CMD=${2}

# Check if the required CGroup exists
find $CGMOUNT -type d | grep $CGP &>/dev/null
if [ $? -ne 0 ]; then
  echo "ERROR: could not find any $CGP cgroup under $CGMOUNT"
  exit 1
fi

find $CGMOUNT -type d | grep $CGP | \
while read CGPATH; do
    # Move this shell into that control group
    echo $$ > $CGPATH/cgroup.procs
    echo "Moving task into $CGPATH"
done

# Execute the command
$CMD
