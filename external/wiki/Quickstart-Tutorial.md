Once you've cloned LISA, source init_env to initialize the LISA Shell, which provides
a convenient set of shell commands for easy access to many LISA related
functions.

```shell
$ source init_env
```

Run `lisa-help` to see an overview of the provided LISA commands.

## Starting the IPython server to use LISA notebooks

To start the IPython Notebook Server required to use this Notebook, on a
LISAShell run:

```shell
[LISAShell lisa] \> lisa-ipython start

Starting IPython Notebooks...
Starting IPython Notebook server...
  IP Address :  http://127.0.0.1:8888/
  Folder     :  /home/derkling/Code/lisa/ipynb
  Logfile    :  /home/derkling/Code/lisa/ipynb/server.log
  PYTHONPATH : 
    /home/derkling/Code/lisa/libs/bart
    /home/derkling/Code/lisa/libs/trappy
    /home/derkling/Code/lisa/libs/devlib
    /home/derkling/Code/lisa/libs/wlgen
    /home/derkling/Code/lisa/libs/utils


Notebook server task: [1] 24745
```

Note that the `lisa-ipython` command allows to specify also interface and
port in case you have several network interfaces on your host:

```lisa-ipython start [interface [port]]```

The URL of the main folder served by the server is printed on the screen.
By default it is 
  http://127.0.0.1:8888/
  
Once the server is started you can have a look at the provide tutorial notebooks
are accessible by following (in your browser) this link:

  http://127.0.0.1:8888/notebooks/tutorial/00_LisaInANutshell.ipynb

This initial tutorial can be seen (but not executed) also on GitHub:

  https://github.com/ARM-software/lisa/blob/master/ipynb/tutorial/00_LisaInANutshell.ipynb

## Running automated tests

To run automated tests, you'll first need to configure the test framework to access your target. 
Edit `target.config` with these details - this file contains comments describing the information 
you'll need to add. For example:

- For an SSH (Linux) target you'll usually need to edit:
  - The "platform" field to "linux"
  - The "board" field to the name of your device (leave blank if it isn't listed as an option)
  - The "host" field to provide the IP address
  - The "username" and "password" fields, or the "keyfile" field to provide login credentials.

- For an ADB (Android) target, you'll usually need to edit:
  - The "platform" to "android"
  - The "board" field to the name of your device (leave blank if it isn't listed as an option)
  - The "device" field to provide the Android device ID.

Once your target is set up, you can run automated tests via the `lisa-test` command in the LISA shell. 
Run `lisa-test help` to see the format of this command.
