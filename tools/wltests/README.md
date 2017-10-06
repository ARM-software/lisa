
# WLTests - Workloads Tests on a Series of Commits

The `lisa-wltest-series` takes a Linux kernel tree, a file containing a list of
commits, and a test command. It then compiles each of those kernels, boots them
on a remote Android target, and runs the test command for each of them.

An IPython notebook is provided for analysing the results.

## Initialization

```bash
# Enter your LISA main folder
$> cd /path/to/your/LISA_HOME

# Initialize a LISAShell
$> source init_env

# Export your ANDROID_HOME
[LISAShell lisa] \> export ANDROID_HOME=/path/to/your/android-sdk-linux

# Ensure your cross-compiler is in your PATH
[LISAShell lisa] \> export PATH=/path/to/your/cross-compiler/bin:$PATH
```
## Prepare the target device

In general your device should be pre-configured and flashed with an updated and
stable user-space. The userspace usually comes with a boot image (`boot.img`)
which provides also a ramdisk iamge. In order to be able to test different
kernels, you are requried to deploy the ramdisk which matches your `boot.img`
under the corresponding platform folder.

For example, if you are targeting an hikey960 board running android-4.4, the
ramdisk image should be deployed under:
```
   tools/wltests/platforms/hikey960_android-4.4/ramdisk.gz
```
Please, note that the name of the ramdisk image, in this example `ramdisk.gz`,
has to match the value for the `RAMDISK_IMAGE` variable defined by the paltform
definition file, in this example:
```
   tools/wltests/platforms/hikey960_android-4.4/definitions
```

### Hikey960
By default, the firmware on that device reports a device ID when in FASTBOOT
mode which is different from the device ID reported when in ADB mode.
This is a major issue for the testing scripts since they required a mandatory
device ID which is expected to be the same in both ADB and FASTBOOT modes.

To fix this, you can set a custom and unique device ID for you hikey960 baord
using the following command from FASTBOOT mode:

```bash
# Set a unique device ID for both FASTBOOT and ADB modes:
[LISAShell lisa] \> DEVICE_ID="UniqueIdYouLike"
[LISAShell lisa] \> fastboot getvar nve:SN@$DEVICE_ID
```

## Download workload dependencies

We cannot distribute the APK files required for this tool to run the workloads -
you will need to do that yourself. You can either install them directly on your
device (from the Play Store, if necessary), or populate
`$LISA_HOME/tools/wa_user_directory/dependencies` so that they can be
automatically installed. There should be one directory for each of the named
workloads, containing the required APK file, like so:

```
[LISAShell lisa] \> tree tools/wa_user_directory/dependencies/
tools/wa_user_directory/dependencies/
├── exoplayer
│   └── exoplayer-demo.apk
└── jankbench
    └── jank-benchmark.apk
```

Note that the leaf filename of the .apk files is not important - the files'
content will be inspected using Android's packaging tools.

#### Exoplayer

Exoplayer can be built from source code. Clone
https://github.com/google/ExoPlayer, open the source tree in Android Studio, and
compile. This should result in a file named 'demo-noExtensions-debug.apk'

## Using the tool

You'll need to create a list of commits that you want to compare the performance
of. This should be a file in the format produced by running
`git log --no-color --oneline` in your kernel tree.

The test command is typically a Workload Automation command - you can use
variable substitution to set the location of the output directory that will be
produced - see the example below.

```bash
# Get a detailed description of the supported options
[LISAShell lisa] \> lisa-wltest-series --help

# Minimal command line to run a Workload Automation agenda
[LISAShell lisa] \> lisa-wltest-series \
			--platform hikey960_android-4.4 \
			--kernel_path /path/to/your/kernel/hikey-linaro \
			--series /path/to/your/series.sha1 \
			--wa_agenda /path/to/your/agenda.yaml
```
