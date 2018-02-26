import sys


class Logger(object):
    """Write messages at the same time in a file and in the terminal"""
    def __init__(self, path):
        """

        :param path: path to the place where we log messages
        """
        self.terminal = sys.stdout
        self.log = open(path + '/hyperband_run.log', 'a')

    def flush(self):
        pass

    def write(self, message):
        """

        :param message: as it is
        :return: None
        """
        self.terminal.write(message)
        self.log.write(message)
        # self.log.flush()
