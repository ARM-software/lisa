import pkgutil

# Import all the submodules before they are asked for by user code, since we
# need to create the *Analysis classes in order for them to be registered
# against AnalysisBase
__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    module = loader.find_module(module_name).load_module(module_name)
    # Load module under its right name, so explicit imports of it will hit the
    # sys.module cache instead of importing twice, with two "version" of each
    # classes defined inside.
    globals()[module_name] = module


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
