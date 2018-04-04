from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from collections import OrderedDict

d_ada = OrderedDict()
d_ada['n_estimators'] = ('int', (5, 200))
d_ada['learning_rate'] = ('cont', (1e-5, 1))


class Ada(object):
    def __init__(self, problem='binary', n_estimators=50, learning_rate=1):
        self.problem = problem
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.name = 'Ada'

    def eval(self):
        if self.problem == 'binary':
            mod = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                     random_state=20)
        else:
            mod = AdaBoostRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                    random_state=20)
        return mod
