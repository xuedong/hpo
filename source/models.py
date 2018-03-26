import abc


class Model:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate_arms(self, n, path, params):
        pass

    @abc.abstractmethod
    def run_solver(self, iterations, arm, classifier):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_search_space():
        pass
