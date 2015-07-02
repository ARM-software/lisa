"""
Default config for Workload Automation. DO NOT MODIFY this file. This file
gets copied to ~/.workload_automation/config.py on initial run of run_workloads.
Add your configuration to that file instead.

"""
#  *** WARNING: ***
# Configuration listed in this file is NOT COMPLETE. This file sets the default
# configuration for WA and gives EXAMPLES of other configuration available. It
# is not supposed to be an exhaustive list.
# PLEASE REFER TO WA DOCUMENTATION FOR THE COMPLETE LIST OF AVAILABLE
# EXTENSIONS AND THEIR CONFIGURATION.


# This defines when the device will be rebooted during Workload Automation execution.              #
#                                                                                                  #
# Valid policies are:                                                                              #
#   never:  The device will never be rebooted.                                                     #
#   as_needed: The device will only be rebooted if the need arises (e.g. if it                     #
#              becomes unresponsive                                                                #
#   initial: The device will be rebooted when the execution first starts, just before executing    #
#            the first workload spec.                                                              #
#   each_spec: The device will be rebooted before running a new workload spec.                     #
#   each_iteration: The device will be rebooted before each new iteration.                         #
#                                                                                                  #
reboot_policy = 'as_needed'

#  Defines the order in which the agenda spec will be executed. At the moment,                     #
#  the following execution orders are supported:                                                   #
#                                                                                                  #
#   by_iteration: The first iteration of each workload spec is executed one ofter the other,       #
#                 so all workloads are executed before proceeding on to the second iteration.      #
#                 This is the default if no order is explicitly specified.                         #
#                 If multiple sections were specified, this will also split them up, so that specs #
#                 in the same section are further apart in the execution order.                    #
#   by_section:   Same as "by_iteration", but runn specs from the same section one after the other #
#   by_spec:      All iterations of the first spec are executed before moving on to the next       #
#                 spec. This may also be specified as ``"classic"``, as this was the way           #
#                 workloads were executed in earlier versions of WA.                               #
#   random:       Randomisizes the order in which specs run.                                       #
execution_order = 'by_iteration'


# This indicates when a job will be re-run.
# Possible values:
#     OK: This iteration has completed and no errors have been detected
#     PARTIAL: One or more instruments have failed (the iteration may still be running).
#     FAILED: The workload itself has failed.
#     ABORTED: The user interupted the workload
#
# If set to an empty list, a job will not be re-run ever.
retry_on_status = ['FAILED', 'PARTIAL']

# How many times a job will be re-run before giving up
max_retries = 3

####################################################################################################
######################################### Device Settings ##########################################
####################################################################################################
# Specify the device you want to run workload automation on. This must be a                        #
# string with the ID of the device. At the moment, only 'TC2' is supported.                        #
#                                                                                                  #
device = 'generic_android'

# Configuration options that will be passed onto the device. These are obviously device-specific,  #
# so check the documentation for the particular device to find out which options and values are    #
# valid. The settings listed below are common to all devices                                       #
#                                                                                                  #
device_config = dict(
    # The name used by adb to identify the device. Use "adb devices" in bash to list
    # the devices currently seen by adb.
    #adb_name='10.109.173.2:5555',

    # The directory on the device that WA will use to push files to
    #working_directory='/sdcard/wa-working',

    # This specifies the device's CPU cores. The order must match how they
    # appear in cpufreq. The example below is for TC2.
    # core_names = ['a7', 'a7', 'a7', 'a15', 'a15']

    # Specifies cluster mapping for the device's cores.
    # core_clusters = [0, 0, 0, 1, 1]
)


####################################################################################################
################################### Instrumention Configuration ####################################
####################################################################################################
# This defines the additionnal instrumentation that will be enabled during workload execution,     #
# which in turn determines what additional data (such as /proc/interrupts content or Streamline    #
# traces) will be available in the results directory.                                              #
#                                                                                                  #
instrumentation = [
    # Records the time it took to run the workload
    'execution_time',

    # Collects /proc/interrupts before and after execution and does a diff.
    'interrupts',

    # Collects the contents of/sys/devices/system/cpu before and after execution and does a diff.
    'cpufreq',

    # Gets energy usage from the workload form HWMON devices
    # NOTE: the hardware needs to have the right sensors in order for this to work
    #'hwmon',

    # Run perf in the background during workload execution and then collect the results. perf is a
    # standard Linux performance analysis tool.
    #'perf',

    # Collect Streamline traces during workload execution. Streamline is part of DS-5
    #'streamline',

    # Collects traces by interacting with Ftrace Linux kernel internal tracer
    #'trace-cmd',

    # Obtains the power consumption of the target device's core measured by National Instruments
    # Data Acquisition(DAQ) device.
    #'daq',

    # Collects CCI counter data.
    #'cci_pmu_logger',

    # Collects FPS (Frames Per Second) and related metrics (such as jank) from
    # the View of the workload (Note: only a single View per workload is
    # supported at the moment, so this is mainly useful for games).
    #'fps',
]


####################################################################################################
################################# Result Processors Configuration ##################################
####################################################################################################
# Specifies how results will be processed and presented.                                           #
#                                                                                                  #
result_processors = [
    # Creates a status.txt that provides a summary status for the run
    'status',

    # Creates a results.txt file for each iteration that lists all collected metrics
    # in "name = value (units)" format
    'standard',

    # Creates a results.csv that contains metrics for all iterations of all workloads
    # in the .csv format.
    'csv',

    # Creates a summary.csv that contains summary metrics for all iterations of all
    # all in the .csv format. Summary metrics are defined on per-worklod basis
    # are typically things like overall scores. The contents of summary.csv are
    # always a subset of the contents of results.csv (if it is generated).
    #'summary_csv',

    # Creates a results.csv that contains metrics for all iterations of all workloads
    # in the JSON format
    #'json',

    # Write results to an sqlite3 database. By default, a new database will be
    # generated for each run, however it is possible to specify a path to an
    # existing DB file (see result processor configuration below), in which
    # case results from multiple runs may be stored in the one file.
    #'sqlite',
]


####################################################################################################
################################### Logging output Configuration ###################################
####################################################################################################
# Specify the format of logging messages. The format uses the old formatting syntax:               #
#                                                                                                  #
#   http://docs.python.org/2/library/stdtypes.html#string-formatting-operations                    #
#                                                                                                  #
# The attributes that can be used in formats are listested here:                                   #
#                                                                                                  #
#   http://docs.python.org/2/library/logging.html#logrecord-attributes                             #
#                                                                                                  #
logging = {
    # Log file format
    'file format': '%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    # Verbose console output format
    'verbose format': '%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    # Regular console output format
    'regular format': '%(levelname)-8s %(message)s',
    # Colouring the console output
    'colour_enabled': True,
}


####################################################################################################
#################################### Instruments Configuration #####################################
####################################################################################################
# Instrumention Configuration is related to specific insturment's settings. Some of the            #
# instrumentations require specific settings in order for them to work. These settings are         #
# specified here.                                                                                  #
# Note that these settings only take effect if the corresponding instrument is
# enabled above.

####################################################################################################
######################################## perf configuration ########################################

# The hardware events such as instructions executed, cache-misses suffered, or branches
# mispredicted to be reported by perf. Events can be obtained from the device by tpying
# 'perf list'.
#perf_events = ['migrations', 'cs']

# The perf options which can be obtained from man page for perf-record
#perf_options = '-a -i'

####################################################################################################
####################################### hwmon configuration ########################################

# The kinds of sensors hwmon instrument will look for
#hwmon_sensors = ['energy', 'temp']

####################################################################################################
###################################### trace-cmd configuration #####################################

# trace-cmd events to be traced. The events can be found by rooting on the device then type
# 'trace-cmd list -e'
#trace_events = ['power*']

####################################################################################################
######################################### DAQ configuration ########################################

# The host address of the machine that runs the daq Server which the insturment communicates with
#daq_server_host = '10.1.17.56'

# The port number for daq Server in which daq insturment communicates with
#daq_server_port = 56788

# The values of resistors 1 and 2 (in Ohms) across which the voltages are measured
#daq_resistor_values = [0.002, 0.002]

####################################################################################################
################################### cci_pmu_logger configuration ###################################

# The events to be counted by PMU
# NOTE: The number of events must not exceed the number of counters available (which is 4 for CCI-400)
#cci_pmu_events = ['0x63', '0x83']

# The name of the events which will be used when reporting PMU counts
#cci_pmu_event_labels = ['event_0x63', 'event_0x83']

# The period (in jiffies) between counter reads
#cci_pmu_period = 15

####################################################################################################
################################### fps configuration ##############################################

# Data points below this FPS will dropped as not constituting "real" gameplay. The assumption
# being that while actually running, the FPS in the game will not drop below X frames per second,
# except on loading screens, menus, etc, which should not contribute to FPS calculation.
#fps_drop_threshold=5

# If set to True, this will keep the raw dumpsys output in the results directory (this is maily
# used for debugging). Note: frames.csv with collected frames data will always be generated
# regardless of this setting.
#fps_keep_raw=False

####################################################################################################
################################# Result Processor Configuration ###################################
####################################################################################################

# Specifies an alternative database to store results in. If the file does not
# exist, it will be created (the directiory of the file must exist however). If
# the file does exist, the results will be added to the existing data set (each
# run as a UUID, so results won't clash even if identical agendas were used).
# Note that in order for this to work, the version of the schema used to generate
# the DB file must match that of the schema used for the current run. Please
# see "What's new" secition in WA docs to check if the schema has changed in
# recent releases of WA.
#sqlite_database = '/work/results/myresults.sqlite'

# If the file specified by sqlite_database exists, setting this to True will
# cause that file to be overwritten rather than updated -- existing results in
# the file will be lost.
#sqlite_overwrite = False

# distribution: internal

####################################################################################################
#################################### Resource Getter configuration #################################
####################################################################################################

# The location on your system where /arm/scratch is mounted. Used by
# Scratch resource getter.
#scratch_mount_point = '/arm/scratch'

# end distribution
