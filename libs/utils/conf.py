
import json
import logging
import os
import re

class JsonConf(object):

    def __init__(self, filename):
        self.filename = filename
        self.json = None

    def load(self):
        """ Parse a JSON file
            First remove comments and then use the json module package
            Comments look like :
                // ...
            or
                /*
                ...
                */
        """
        if not os.path.isfile(self.filename):
            raise RuntimeError(
                'Missing configuration file: {}'.format(self.filename)
            )
        logging.debug('loading JSON...')

        with open(self.filename) as fh:
            content = ''.join(fh.readlines())

            ## Looking for comments
            match = JSON_COMMENTS_RE.search(content)
            while match:
                # single line comment
                content = content[:match.start()] + content[match.end():]
                match = JSON_COMMENTS_RE.search(content)

            # Return json file
            self.json = json.loads(content, parse_int=int)
            logging.debug('Target config: %s', self.json)

        return self.json

    def show(self):
        print json.dumps(self.json, indent=4)

# Regular expression for comments
JSON_COMMENTS_RE = re.compile(
    '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)


