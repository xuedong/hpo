from sklearn.svm import SVC, SVR
from collections import OrderedDict

d_svm = OrderedDict()
d_svm['C'] = ('cont', (-4, 5))
d_svm['gamma'] = ('cont', (-4, 5))


class SVM(object):
    def __init__(self, problem='binary', c=0, gamma=0, kernel='rbf'):
        self.problem = problem
        self.C = 10 ** c
        self.gamma = 10 ** gamma
        self.kernel = kernel
        self.name = 'SVM'

    def eval(self):
        if self.problem == 'binary':
            mod = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True, random_state=20)
        else:
            mod = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma)
        return mod
