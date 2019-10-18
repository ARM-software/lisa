
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

# Export your ANDROID_HOME if it not set already to the correct path
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

The board should also be configure with a valid Internet connection.

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

If the tool finds that an .apk file is installed on the device, but not present
on the host, it will be pulled into your dependencies/ directory.

#### Exoplayer

Exoplayer is the underlying tech used by the YouTube Android app. The hope is
that it can be used as a proxy for Youtube performance on devices where running
Youtube itself is not practical.

Exoplayer can be built from source code. Clone
https://github.com/google/ExoPlayer, open the source tree in Android Studio, and
compile. This should result in a file named 'demo-noExtensions-debug.apk'.

#### Jankbench

You'll need to get the Jankbench .apk from Google.

#### YouTube

By its nature, YouTube needs to be pre-installed on the device for the
automation to work. Note that WA3 has two YouTube workloads: The "youtube"
workload simulates UI interactions, while the "youtube_playback" simply plays a
video from a URL. The former workload appears to be susceptible to
reproducibility issues as the content that is rendered (such as advertisements
and video recommendations) can change between invocations.

#### Geekbench

The Geekbench automation should be pretty robust. The easiest way to get hold of
it is probably just to install it from the Play Store.

Geekbench requires Internet access to run. Thus, you need to ensure your device
is properly configured for Internet access before running this test.
On Hikey960, for example, this requires to temporarily connect a keyboard and
mouse to your device to setup a WiFi connection. Once Internet access has been
configured then you need to remove keyboard and mouse and plug in the USB
Type-C for ADB to work.

Note that as Geekbench poses a threat of 'phoning home', the tool marks it as
dangerous. The WA3 configuration file provided with this tool in
   `$LISA_HOME/tools/wa_user_directory/config.yaml`
sets
   `allow_phone_home: false`

This allows to prevent accidentally running Geekbench on a confidential
device. Therefore you will need to have that setting if you are running
on a sensible device.
If you don't have any confidential devices you can use the default
configuration.
Otherwise, it is best to create a separate per-device config file that
overrides it, for example:

```
$ cat hikey960-config.yaml
device_config:
  device: 4669290103000000

allow_phone_home: false
```

Adding `-c /path/to/hikey960config.yaml` to the `wa` command will apply this
configuration.

#### PCMark

The PCMark automation support in this tool is very limited. You'll need to
manually install the .apk from
http://www.futuremark.com/downloads/pcmark-android.apk, open it on the device
and hit the 'install' button to install the 'Work' benchmark.
Note that an Internet connection is required to complete the installation.
Furthermore, the robustness of the UI automation is not up to the standards of
the other workloads in WA, so there may be issues running it on untested
devices.
A proper solution would require writing UiAutomator code in the vein of WA's
[Vellamo workload](https://github.com/ARM-software/workload-automation/blob/next/wa/workloads/vellamo/uiauto/app/src/main/java/com/arm/wa/uiauto/vellamo/UiAutomation.java).
Part of the reason this hasn't been done is that PCMark displays its content in
a WebView, which poses a challenge for automation with Android's API.

## Using the tool

### Running workloads using different kernels

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
			--kernel_src /path/to/your/kernel/hikey-linaro \
			--series /path/to/your/series.sha1 \
			--wa_agenda /path/to/your/agenda.yaml \
			--device <ADB_DEVICE_ID>
```

*NOTE:* a complete set of results should include energy measurements.
To collect energy measurements, wltests supports out-of-the box the
[ACME EnergyMeter](https://github.com/ARM-software/lisa/wiki/Energy-Meters-Requirements#iiocapture---baylibre-acme-cape).
For example, to collect energy measurements from channels 1 and 2, you can just add:
`--acme_channels "1 2"` to the above command.

For a more complete list of configuration options, please have a look at the output of:
```bash
[LISAShell lisa] \> lisa-wltest-series --help
```

### Collecting and comparing results from different kernels

Once the workloads specified by the agenda have been executed for all the
kernels specified by the series file, collected results can be post-processed
and analyzed using a wltests companion notebook usually available under:

   https://github.com/ARM-software/lisa/blob/master/ipynb/wltests

An example of the report produced when comparing a kernel using WALT versus a
kernel using PELT on Hikey960 is available here:

   https://gist.github.com/derkling/3a8c3568676a29e608d6dcb15af06241


## Using the tool to evaluate scheduler patches

One of the main usages of the wltest suite is to compare the power and
performance benefits/overheads for a set of proposed scheduler patches.
To this purposed, wltests provides out-of-the box support to run a
representative set of Android workloads on an hikey960 board while measuring
energy using an
[ACME Energymeter](https://github.com/ARM-software/lisa/wiki/Energy-Meters-Requirements#iiocapture---baylibre-acme-cape).

For this kind of evaluation, please use:
- agenda: [tools/wltests/agendas/sched-evaluation-full.yaml](https://github.com/ARM-software/lisa/blob/master/tools/wltests/agendas/sched-evaluation-full.yaml)
- notebook: [ipynb/wltests/sched-evaluation-full.ipynb](https://github.com/ARM-software/lisa/blob/master/ipynb/wltests/sched-evaluation-full.ipynb)

