import abc


class Model:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def generate_arms(n, path, params):
        pass

    @staticmethod
    @abc.abstractmethod
    def run_solver(iterations, arm, data):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_search_space():
        pass
