class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class EventLogger:
    def __init__(self, bo):
        self.bo = bo
        self.header = 'Evaluation \t Proposed point \t  Current eval. \t Best eval.'
        self.template = '{:6} \t {}. \t  {:6} \t {:6}'
        print(self.header)

    def _printCurrent(self, bo):
        eval = str(len(bo.GP.y) - bo.init_evals)
        proposed = str(bo.best)
        curr_eval = str(bo.GP.y[-1])
        curr_best = str(bo.tau)
        if float(curr_eval) >= float(curr_best):
            curr_eval = bcolors.OKGREEN + curr_eval + bcolors.ENDC
        print(self.template.format(eval, proposed, curr_eval, curr_best))

    def _printInit(self, bo):
        for init_eval in range(bo.init_evals):
            print(self.template.format('init', bo.GP.X[init_eval], bo.GP.y[init_eval], bo.tau))

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from bo.covfunc import squaredExponential
    from bo.surrogates.GaussianProcess import GaussianProcess
    from bo.acquisition import Acquisition
    from bo.GPGO import GPGO

    np.random.seed(20)

    def f(x):
        return -((6*x-2)**2*np.sin(12*x-4))

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode = 'ExpectedImprovement')

    params = {'x': ('cont', (0, 1))}
    bo = BO(gp, acq, f, params)
    bo.run(max_iter = 10)
    print(bo.getResult())
