import logging


class TraceCollector(object):

    def __init__(self, target):
        self.target = target
        self.logger = logging.getLogger(self.__class__.__name__)

    def reset(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def get_trace(self, outfile):
        pass
