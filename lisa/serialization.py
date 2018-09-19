from ruamel.yaml import YAML
import copy

class YAMLSerializable:
    serialized_whitelist = []
    serialized_blacklist = []
    serialized_placeholders = dict()

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

    #TODO: figure out why it is breaking deserialization of circular object
    # graphs
    #  def __getstate__(self):
        #  """
        #  Filter the instance's attributes upon serialization.

        #  The following class attributes can be used to customize the serialized
        #  content:
            #  * :attr:`serialized_whitelist`: list of attribute names to
              #  serialize. All other attributes will be ignored and will not be
              #  saved/restored.

            #  * :attr:`serialized_blacklist`: list of attribute names to not
              #  serialize.  All other attributes will be saved/restored.

            #  * serialized_placeholders: Map of attribute names to placeholder
              #  values. These attributes will not be serialized, and the
              #  placeholder value will be used upon restoration.

            #  If both :attr:`serialized_whitelist` and
            #  :attr:`serialized_blacklist` are specified,
            #  :attr:`serialized_blacklist` is ignored.
        #  """

        #  dct = copy.copy(self.__dict__)
        #  if self.serialized_whitelist:
            #  dct = {attr: dct[attr] for attr in self.serialized_whitelist}

        #  elif self.serialized_blacklist:
            #  for attr in self.serialized_blacklist:
                #  dct.pop(attr, None)

        #  for attr, placeholder in self.serialized_placeholders.items():
            #  dct.pop(attr, None)

        #  return dct

    #  def __setstate__(self, dct):
        #  if self.serialized_placeholders:
            #  dct.update(copy.deepcopy(self.serialized_placeholders))
        #  self.__dict__ = dct

    def __copy__(self):
        """Make sure that copying the class still works as usual, without
        dropping some attributes.
        """
        return super().__copy__(self)

    def __deepcopy__(self):
        return super().__deepcopy__()

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
