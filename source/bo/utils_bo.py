import numpy as np

from bo.bo import BO
from bo.surrogates.gaussian_process import GaussianProcess
from bo.acquisition import Acquisition
from bo.covfunc import SquaredExponential
from utils import build, Loss, cum_max


def evaluate_dataset(csv_path, target_index, problem, model, parameter_dict, method='holdout', seed=20, max_iter=50):
    print('Now evaluating {}...'.format(csv_path))
    x, y = build(csv_path, target_index)

    wrapper = Loss(model, x, y, method=method, problem=problem)

    # print('Evaluating PI')
    # np.random.seed(seed)
    # sexp = SquaredExponential()
    # gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    # acq_pi = Acquisition(mode='probability_improvement')
    # bo_pi = BO(gp, acq_pi, wrapper.evaluate_loss, parameter_dict, n_jobs=1)
    # bo_pi.run(max_iter=max_iter)

    print('Evaluating EI')
    np.random.seed(seed)
    sexp = SquaredExponential()
    gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    acq_ei = Acquisition(mode='expected_improvement')
    bo_ei = BO(gp, acq_ei, wrapper.evaluate_loss, parameter_dict, n_jobs=1)
    bo_ei.run(max_iter=max_iter)

    # Also add gpucb, beta = 0.5, beta = 1.5
    print('Evaluating GP-gpucb beta = 0.5')
    np.random.seed(seed)
    sexp = SquaredExponential()
    gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    acq_ucb = Acquisition(mode='gpucb', beta=0.5)
    bo_ucb = BO(gp, acq_ucb, wrapper.evaluate_loss, parameter_dict, n_jobs=1)
    bo_ucb.run(max_iter=max_iter)

    # print('Evaluating GP-gpucb beta = 1.5')
    # np.random.seed(seed)
    # sexp = SquaredExponential()
    # gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    # acq_ucb2 = Acquisition(mode='gpucb', beta=1.5)
    # bo_ucb2 = BO(gp, acq_ucb2, wrapper.evaluate_loss, parameter_dict, n_jobs=1)
    # bo_ucb2.run(max_iter=max_iter)

    print('Evaluating random')
    np.random.seed(seed)
    r = evaluate_random(bo_ei, wrapper.evaluate_loss, n_eval=max_iter + 1)
    r = cum_max(r)

    # pi_h = np.array(gpgo_pi.history)
    ei_h = np.array(bo_ei.history)
    ucb1_h = np.array(bo_ucb.history)
    # ucb2_h = np.array(gpgo_ucb2.history)

    return ei_h, ucb1_h, r


def evaluate_random(bo_model, loss, n_eval=20):
    res = []
    for i in range(n_eval):
        param = bo_model.sample_param()
        current_loss = loss(**param)
        res.append(current_loss)
        print('Param {}, Loss: {}'.format(param, current_loss))
    return res
