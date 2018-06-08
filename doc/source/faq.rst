.. _faq:

FAQ
===

.. contents::
   :depth: 1
   :local:

---------------------------------------------------------------------------------------


**Q:** I receive the error: ``"<<Workload> file <file_name> file> could not be found."``
-----------------------------------------------------------------------------------------

**A:** Some workload e.g. AdobeReader, GooglePhotos etc require external asset
files. We host some additional workload dependencies in the `WA Assets Repo
<https://github.com/ARM-software/workload-automation-assets>`_. To allow WA to
try and automatically download required assets from the repository please add
the following to your configuration:

.. code-block:: YAML

        remote_assets_url: https://raw.githubusercontent.com/ARM-software/workload-automation-assets/master/dependencies

------------

**Q:** I receive the error: ``"No matching package found for workload <workload>"``
------------------------------------------------------------------------------------

**A:** WA cannot locate the application required for the workload. Please either
install the application onto the device or source the apk and place into
``$WA_USER_DIRECTORY/dependencies/<workload>``

------------

**Q:** I am trying to set a valid runtime parameters however I still receive the error ``"Unknown runtime parameter"``
-------------------------------------------------------------------------------------------------------------------------

**A:** Please ensure you have the corresponding module loaded on the device.
See :ref:`Runtime Parameters <runtime-parameters>` for the list of
runtime parameters and their containing modules, and the appropriate section in
:ref:`setting up a device <setting-up-a-device>` for ensuring it is installed.

-------------

**Q:** I have a big.LITTLE device but am unable to set parameters corresponding to the big or little core and receive the error ``"Unknown runtime parameter"``
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

**A:** Please ensure you have the hot plugging module enabled for your device (Please see question above).


**A:** This can occur if the device uses dynamic hot-plugging and although WA
will try to online all cores to perform discovery sometimes this can fail
causing to WA to incorrectly assume that only one cluster is present. To
workaround this please set the ``core_names`` :ref:`parameter <core-names>` in the configuration for
your device.


**Q:** I receive the error ``Could not find plugin or alias "standard"``
------------------------------------------------------------------------

**A:** Upon first use of WA3, your WA2 config file typically located at
``$USER_HOME/config.py`` will have been converted to a WA3 config file located at
``$USER_HOME/config.yaml``. The "standard" output processor, present in WA2, has
been merged into the core framework and therefore no longer exists. To fix this
error please remove the "standard" entry from the "augmentations" list in the
WA3 config file.

**Q:** My Juno board keeps resetting upon starting WA even if it hasn't crashed.
--------------------------------------------------------------------------------
Please ensure that you do not have any other terminals (e.g. ``screen``
sessions) connected to the board's UART. When WA attempts to open the connection
for its own use this can cause the board to reset if a connection is already
present.
