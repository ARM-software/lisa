## Systrace

When performing tests on Android devices, it is possible to exploit [`systrace`](https://developer.android.com/studio/profile/systrace-commandline.html).

In order to use `systrace` it is recommended to set up `catapult` by simply cloning the repository:

    $ git clone https://github.com/catapult-project/catapult

and use `systrace` inside an IPython Notebook. Examples of usage of `systrace` can be found in:

* [Android_YouTube](https://github.com/ARM-software/lisa/blob/master/ipynb/android/workloads/Android_YouTube.ipynb)
* [Android_Workloads](https://github.com/ARM-software/lisa/blob/master/ipynb/android/Android_Workloads.ipynb)

### atrace

`atrace` is an Android binary called by `systrace` via `adb` to capture kernel events
using `ftrace`. 