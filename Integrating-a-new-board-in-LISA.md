Adding support for your device is a simple as writing a JSON file with the following information:

```json
{
    "board" : {
        "cores" : ["A53", "A53", "A57", "A57"],
        "big_core" : "A57",
        "modules" : ["bl", "cpufreq"]
    }
}
```
Where:

- `cores`: is the list of core names of the cores available in the system
- `big_core`: is the name of the big core (must be one of the names 
- `modules`: (optional) the list of `devlib` modules to be loaded by default for this board

The two parameters are needed to understand the topology of the target device.

The file must be placed under `<LISA_HOME>/utils/platforms/`.

NOTE: As of now, **only dual cluster devices are fully supported**.
To add a single-cluster device, remove the `big_core` entry and the `bl` module
from the json configuration.

