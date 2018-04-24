import math
import sys
# import numpy as np
import timeit
from six.moves import cPickle

import source.target as target
import source.ho.utils_ho as utils_ho
import log.logger as logger
# import source.classifiers.logistic as logistic
from source.classifiers.mlp_sklearn import *
from source.classifiers.svm_sklearn import *
# from source.classifiers.tree_sklearn import *
# from source.classifiers.rf_sklearn import *
from source.classifiers.knn_sklearn import *
from source.classifiers.gbm_sklearn import *
from source.classifiers.ada_sklearn import *


if __name__ == '__main__':
    horizon = 10
    mcmc = 1
    rho = 0.66
    nu = 1.
    sigma = 0.1
    delta = 0.05
    alpha = math.log(horizon) * (sigma ** 2)
    c = 2 * math.sqrt(1. / (1 - 0.66))
    c1 = (0.66 / (3 * 1.)) ** (1. / 8)
    # f_target = target.Rosenbrock(1, 100)
    # bbox = utils_ho.std_box(f_target.f, f_target.fmax, 2, 0.1, [(-3., 3.), (-3., 3.)], ['cont', 'cont'])
    #
    # regrets = np.zeros(horizon)
    # for k in range(10):
    #     current, _, _ = utils_ho.regret_hct(bbox=bbox, rho=0.66, nu=1., c=c, c1=c1, delta=0.05, horizon=horizon)
    #     regrets = np.add(regrets, current)
    #     # print(current)
    # x = range(horizon)
    # plt.plot(x, regrets/10.)
    # plt.show()

    # data_dir = 'mnist.pkl.gz'
    # data = utils.load_data(data_dir)
    #
    # start = timeit.default_timer()
    #
    # # x = ts.matrix('x')
    # # test_model = logistic.LogisticRegression(x, 28*28, 10)
    # params = logistic.LogisticRegression.get_search_space()
    #
    # f_target = target.TheanoLogistic(1, data, ".")
    # bbox = utils_ho.std_box(f_target, None, 2, 0.1,
    #                         [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
    #                          (params['batch_size'].get_min(), params['batch_size'].get_max())],
    #                         [params['learning_rate'].get_type(), params['batch_size'].get_type()])
    #
    # current = [0. for _ in range(horizon)]
    # alpha = math.log(10) * (0.1 ** 2)
    # losses = utils_ho.loss_hct(bbox=bbox, rho=0.66, nu=1., c=c, c1=c1, delta=0.05, sigma=0.1, horizon=horizon)
    # # losses = utils_ho.loss_hoo(bbox=bbox, rho=0.66, nu=1., alpha=alpha, sigma=0.1, horizon=10, update=False)
    # print(losses)

    models = [Ada()]
    model_names = ['ada_']
    # models = [Ada(), KNN(), MLP(), GBM(), SVM()]
    # model_names = ['ada_', 'knn_', 'sk_mlp_', 'gbm_', 'svm_']
    targets = [target.SklearnAda, target.SklearnKNN, target.SklearnMLP, target.SklearnGBM, target.SklearnSVM]
    params = [d_ada, d_knn, d_mlp, d_gbm, d_svm]
    path = os.path.join(os.getcwd(), '../data/uci')
    dataset = 'breast_cancer.csv'
    problem = 'binary'
    target_index = 0
    x, y = utils.build(os.path.join(path, dataset), target_index)
    output_dir = ''
    # rng = np.random.RandomState(12345)

    for i in range(len(models)):
        model = models[i]
        model_name = model_names[i]
        target_class = targets[i]
        param = params[i]

        print('<-- Running HOO -->')
        exp_name = 'hoo_' + model_name + '1/'
        for seed_id in range(mcmc):
            director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
            if not os.path.exists(director):
                os.makedirs(director)
            log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            sys.stdout = logger.Logger(log_dir, 'hoo')

            start_time = timeit.default_timer()

            f_target = target_class(model, x, y, '5fold', problem, director)
            bbox = utils_ho.std_box(f_target, None, 2, 0.1,
                                    [param[key][1] for key in param.keys()],
                                    [param[key][0] for key in param.keys()])
            losses = utils_ho.loss_hoo(bbox=bbox, rho=rho, nu=nu, alpha=alpha, sigma=sigma,
                                       horizon=horizon, director=director, update=False)
            losses = np.array(losses)

            with open(director + '/results.pkl', 'wb') as file:
                cPickle.dump(-losses, file)

            end_time = timeit.default_timer()

            print(('The code for the trial number ' +
                   str(seed_id) +
                   ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

        print('<-- Running HCT -->')
        exp_name = 'hct_' + model_name + '1/'
        for seed_id in range(mcmc):
            director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
            if not os.path.exists(director):
                os.makedirs(director)
            log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            sys.stdout = logger.Logger(log_dir, 'hct')

            start_time = timeit.default_timer()

            f_target = target_class(model, x, y, '5fold', problem, director)
            bbox = utils_ho.std_box(f_target, None, 2, 0.1,
                                    [param[key][1] for key in param.keys()],
                                    [param[key][0] for key in param.keys()])
            losses = utils_ho.loss_hct(bbox=bbox, rho=rho, nu=nu, c=c, c1=c1, delta=delta, sigma=sigma,
                                       horizon=horizon, director=director)
            losses = np.array(losses)

            with open(director + '/results.pkl', 'wb') as file:
                cPickle.dump(-losses, file)

            end_time = timeit.default_timer()
