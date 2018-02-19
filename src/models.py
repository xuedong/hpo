import abc


class Model:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate_arms(self, n, path):
        pass

    @abc.abstractmethod
    def run_solver(self, iterations, arm):
        pass
