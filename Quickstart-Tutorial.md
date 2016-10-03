Once cloned, source init_env to initialized the LISA Shell, which provides
a convenient set of shell commands for easy access to many LISA related
functions.

```shell
$ source init_env
```

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

The main folder served by the server is:
  http://127.0.0.1:8888/
  
Once the server is started you can have a look at the provide tutorial notebooks
are accessible by following (in your browser) this link:

  http://127.0.0.1:8888/notebooks/tutorial/00_LisaInANutshell.ipynb

This intial tutorial can be seen (but not executed) also on GitHub:

  https://github.com/ARM-software/lisa/blob/master/ipynb/tutorial/00_LisaInANutshell.ipynb
