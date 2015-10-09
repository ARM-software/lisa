import logging


class Platform(object):

    @property
    def number_of_clusters(self):
        return len(set(self.core_clusters))

    def __init__(self,
                 name=None,
                 core_names=None,
                 core_clusters=None,
                 big_core=None,
                 model=None,
                 modules=None,
                 ):
        self.name = name
        self.core_names = core_names or []
        self.core_clusters = core_clusters or []
        self.big_core = big_core
        self.little_core = None
        self.model = model
        self.modules = modules or []
        self.logger = logging.getLogger(self.name)
        if not self.core_clusters and self.core_names:
            self._set_core_clusters_from_core_names()
        self._validate()

    def init_target_connection(self, target):
        # May be ovewritten by subclasses to provide target-specific
        # connection initialisation.
        pass

    def update_from_target(self, target):
        if not self.core_names:
            self.core_names = target.cpuinfo.cpu_names
            self._set_core_clusters_from_core_names()
        if not self.big_core and self.number_of_clusters == 2:
            big_idx = self.core_clusters.index(max(self.core_clusters))
            self.big_core = self.core_names[big_idx]
        if not self.core_clusters and self.core_names:
            self._set_core_clusters_from_core_names()
        if not self.model:
            self._set_model_from_target(target)
        if not self.name:
            self.name = self.model
        self._validate()

    def _set_core_clusters_from_core_names(self):
        self.core_clusters = []
        clusters = []
        for cn in self.core_names:
            if cn not in clusters:
                clusters.append(cn)
            self.core_clusters.append(clusters.index(cn))

    def _set_model_from_target(self, target):
        if target.os == 'android':
            self.model = target.getprop('ro.product.model')
        elif target.is_rooted:
            try:
                self.model = target.execute('dmidecode -s system-version',
                                            as_root=True).strip()
            except Exception:  # pylint: disable=broad-except
                pass  # this is best-effort

    def _validate(self):
        if len(self.core_names) != len(self.core_clusters):
            raise ValueError('core_names and core_clusters are of different lengths.')
        if self.big_core and self.number_of_clusters != 2:
            raise ValueError('attempting to set big_core on non-big.LITTLE device. '
                             '(number of clusters  is not 2)')
        if self.big_core and self.big_core not in self.core_names:
            message = 'Invalid big_core value "{}"; must be in [{}]'
            raise ValueError(message.format(self.big_core,
                                            ', '.join(set(self.core_names))))
        if self.big_core:
            little_idx = self.core_clusters.index(min(self.core_clusters))
            self.little_core = self.core_names[little_idx]

