import numpy as np
import sys
import timeit
import matplotlib.pyplot as plt

import source.utils as utils

import source.classifiers.logistic as logistic
# import source.classifiers.mlp as mlp
# import source.baseline.random_search as random_search


if __name__ == '__main__':
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    # rng = np.random.RandomState(12345)
    model_name = 'logistic_sgd_'

    test_model = logistic.LogisticRegression
    params = logistic.LogisticRegression.get_search_space()

    tracks = np.array([None for _ in range(10)])
    for seed_id in range(10):
        start_time = timeit.default_timer()

        arm = {'dir': '.'}
        hps = ['learning_rate', 'batch_size']
        for hp in hps:
            val = params[hp].get_param_range(1, stochastic=True)
            arm[hp] = val[0]
        arm['results'] = []
        train_loss, val_err, test_err, track = \
            test_model.run_solver(100, arm, data, verbose=True)
        tracks[seed_id] = track

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

    x = range(len(tracks[0]))
    y = np.mean(tracks, axis=0)
    plt.plot(x, y)

    plt.grid()
    plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Epochs')
    plt.show()
