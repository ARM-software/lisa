## How do I get a new device up and running in LISA ?

As it stands, there is a bit of work to get a new target device working with LISA. There are plans to make this process more user-friendly, see https://github.com/ARM-software/lisa/issues/425

### Platform description

To run some post-processing, LISA will be looking for a platform description found in **\${LISA_HOME}/libs/utils/platforms/<board-name\>.json**. This board name is either referenced in the Notebook you are running, or in the **${LISA_HOME}/target.config** file. That configuration should have these fields:

```
my_conf = {
    # Target platform and board
    "platform"    : 'android',
    "board"    : '<board-name>',
    ...
}
```

LISA can read the energy-model from the device itself, but the platform file is currently required. As such, you need to generate it. [This branch](https://github.com/valschneider/lisa/tree/better_nrg) has two commits to dump the platform file after reading the energy model from the device, but you should make sure the little and big clusters are labelled correctly.

Furthermore, you will have to to manually fill-in the core info (`"board"` dictionary entry). See **${LISA_HOME}/libs/utils/platforms/pixel.json** for an example of such an entry.