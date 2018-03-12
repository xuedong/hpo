class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class EventLogger:
    def __init__(self, bo_instance):
        self.bo = bo_instance
        self.header = 'Evaluation \t Proposed point \t  Current eval. \t Best eval.'
        self.template = '{:6} \t {}. \t  {:6} \t {:6}'
        print(self.header)

    def _print_current(self, bo_instance):
        evaluation = str(len(bo_instance.GP.y) - bo_instance.init_evals)
        proposed = str(bo_instance.best)
        curr_eval = str(bo_instance.GP.y[-1])
        curr_best = str(bo_instance.tau)
        if float(curr_eval) >= float(curr_best):
            curr_eval = BColors.OKGREEN + curr_eval + BColors.ENDC
        print(self.template.format(evaluation, proposed, curr_eval, curr_best))

    def _print_init(self, bo_instance):
        for init_eval in range(bo_instance.init_evals):
            print(self.template.format('init', bo_instance.GP.X[init_eval],
                                       bo_instance.GP.y[init_eval], bo_instance.tau))


if __name__ == '__main__':
    import numpy as np
    # import matplotlib.pyplot as plt
    from bo.covfunc import SquaredExponential
    from bo.surrogates.gaussian_process import GaussianProcess
    from bo.acquisition import Acquisition
    from bo.bo import BO

    np.random.seed(20)

    def f(x):
        return -((6*x-2)**2*np.sin(12*x-4))

    sexp = SquaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')

    params = {'x': ('cont', (0, 1))}
    bo = BO(gp, acq, f, params)
    bo.run(max_iter=10)
    print(bo.getResult())
