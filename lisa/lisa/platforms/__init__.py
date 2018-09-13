import json
import os
from lisa.conf import JsonConf

#TODO: remove that

# Add identifiers for each of the platforms we have JSON descriptors for
this_dir = os.path.dirname(__file__)
for file_name in os.listdir(this_dir):
    name, ext = os.path.splitext(file_name)
    if ext == '.json':
        platform = JsonConf(os.path.join(this_dir, file_name)).load()
        identifier = name.replace('-', '_')
        globals()[identifier] = platform
