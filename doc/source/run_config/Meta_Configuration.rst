user_directory:  
    type: ``'str'``

    Path to the user directory. This is the location WA will look for
    user configuration, additional plugins and plugin dependencies.

    default: ``'~/.workload_automation'``

assets_repository:  
    type: ``'str'``

    The local mount point for the filer hosting WA assets.

logging:  
    type: ``'LoggingConfig'``

    WA logging configuration. This should be a dict with a subset
    of the following keys::

        :normal_format: Logging format used for console output
        :verbose_format: Logging format used for verbose console output
        :file_format: Logging format used for run.log
        :color: If ``True`` (the default), console logging output will
                contain bash color escape codes. Set this to ``False`` if
                console output will be piped somewhere that does not know
                how to handle those.

    default: ::

        {
            color: True,
            verbose_format: %(asctime)s %(levelname)-8s %(name)s: %(message)s,
            regular_format: %(levelname)-8s %(message)s,
            file_format: %(asctime)s %(levelname)-8s %(name)s: %(message)s
        }

verbosity:  
    type: ``'integer'``

    Verbosity of console output.

default_output_directory:  
    type: ``'str'``

    The default output directory that will be created if not
    specified when invoking a run.

    default: ``'wa_output'``

extra_plugin_paths:  
    type: ``'list_of_strs'``

    A list of additional paths to scan for plugins.

