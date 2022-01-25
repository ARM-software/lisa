#! /usr/bin/env python3

import warnings
import os

# Widespread distutils warning that create noise during tests and would prevent
# treating warnings as errors.
warnings.filterwarnings(
    action='ignore',
    message=r'.*distutils Version classes are deprecated.*',
    category=DeprecationWarning,
)

# Prevent matplotlib from trying to connect to X11 server, for headless testing.
# Must be done before importing matplotlib.pyplot or pylab
try:
    import matplotlib
except ImportError:
    pass
else:
    if not os.getenv('DISPLAY'):
        matplotlib.use('Agg')
