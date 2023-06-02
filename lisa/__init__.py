#! /usr/bin/env python3

import warnings
import os

########################################################################
# Note: imports must be limited to the maximum here, and under no circumstances
# import a package that creates a background thread at import time. Failure to
# comply will prevent lisa._unshare._do_unshare() to work correctly, as it
# cannot work if the process is multithreaded when it is called.
########################################################################


from lisa.version import __version__

warnings.warn('This code was obtained by cloning the "master" git branch of LISA project. The project has moved to the "main" branch. The "master" branch will be tracking the "main" branch for a while. Please update your workflow to use the "main" branch', category=FutureWarning)

# Raise an exception when a deprecated API is used from within a lisa.*
# submodule. This ensures that we don't use any deprecated APIs internally, so
# they are only kept for external backward compatibility purposes.
warnings.filterwarnings(
    action='error',
    category=DeprecationWarning,
    module=fr'{__name__}\..*',
)

try:
    from cryptography.utils import CryptographyDeprecationWarning
except ImportError:
    pass
else:
    warnings.filterwarnings(
        action='ignore',
        category=CryptographyDeprecationWarning,
    )

# When the deprecated APIs are used from __main__ (script or notebook), always
# show the warning
warnings.filterwarnings(
    action='always',
    category=DeprecationWarning,
    module=r'__main__',
)

# Work around the warnings reported here:
# https://github.com/pola-rs/polars/issues/20000
os.environ.setdefault('POLARS_ALLOW_FORKING_THREAD', '1')

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
