from ruamel.yaml import YAML

class YAMLSerializable:

    _yaml = YAML(typ='unsafe')
    _yaml.allow_unicode = True

    def to_stream(self, stream):
        self._yaml.dump(self, stream)

    @classmethod
    def from_stream(cls, stream):
        return cls._yaml.load(stream)

    def to_path(self, filepath):
        #TODO: check if encoding='utf-8' would be beneficial
        with open(filepath, "w") as fh:
            self.to_stream(fh)

    @classmethod
    def from_path(cls, filepath):
        with open(filepath, "r") as fh:
            return cls.from_stream(fh)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
