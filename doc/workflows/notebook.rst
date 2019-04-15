*********
Notebooks
*********

Starting the server
===================

The LISA shell can simplify starting an Jupyter notebook server::

  [LISAShell lisa] \> lisa-jupyter start

  Starting Jupyter Notebooks...

  Notebook server configuration:
    URL        :  http://127.0.0.1:8888/?token=b34F8D0e457BDa570C4A6D7AF113CB45d9CcAF44Aa7Cf400
    Folder     :  /data/work/lisa/ipynb
    Logfile    :  /data/work/lisa/ipynb/server.log
    PYTHONPATH :
	  /data/work/lisa/modules_root/


  Notebook server task: [4] 30177

Note that the ``lisa-jupyter`` command allows you to specify interface and
port in case you have several network interfaces on your host::

  lisa-jupyter start [interface [port]]

The URL of the main folder served by the server is printed on the screen.
By default it is http://127.0.0.1:8888/.

Once the server is started you can have a look at the provided tutorial
notebooks are accessible by following this `link
<http://127.0.0.1:8888/notebooks/examples/typical_experiment.ipynb>`__.
This initial tutorial can be seen (but not executed) also on `github
<https://github.com/ARM-software/lisa/blob/next/ipynb/examples/typical_experiment.ipynb>`__.

Notebooks as development environment
====================================

.. tip::

  To avoid having to restart the kernel and re-import LISA modules that you
  have changed (e.g. you're coding some new feature and testing it out in a
  notebook), you can add this in the first cell of your notebook::

     %load_ext autoreload
     %autoreload 2

  Note that this can cause a few type checking issues, but you should get an
  explicit error in that case.
