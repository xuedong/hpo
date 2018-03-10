import sys


class Logger(object):
    """Write messages at the same time in a file and in the terminal"""
    def __init__(self, path, name):
        """

        :param path: path to the place where we log messages
        """
        self.terminal = sys.stdout
        self.name = name
        self.log = open(path + '/' + self.name + '_run.log', 'a')

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
