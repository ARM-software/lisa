from lisa.version import __version__

# Put all the code inside one function, to allow easy cleanup of the namespace
# at the end. We definitely don't want to expose these things to the outside
# world
def f():
    import os
    if os.getenv('LISA_DO_NOTEBOOK_SETUP'):
        from lisa.utils import jupyter_notebook_setup
        jupyter_notebook_setup()

f()
del f


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
