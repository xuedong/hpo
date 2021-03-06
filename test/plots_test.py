import source.plots as plots


if __name__ == "__main__":
    # path = "../result/hyperband_logistic_0/"
    # plots.plot_hyperband(path, 3, 10, 'logistic_', 'sgd_', 'mnist', 0)
    # path = "../result/hyperband_mlp_0/"
    # plots.plot_hyperband(path, 3, 5, 'mlp_', 'sgd_', 'mnist', 0)
    # path = "../result/hyperband_logistic_2"
    # plots.plot_hyperband(path, 0, 5, 'logistic_', 'sgd_', 'mnist', 2)

    # path = "../result/random_logistic_0"
    # plots.plot_random(path, 1, 'logistic_', 'sgd_', 'mnist', 0)

    # path = "../result/hyperband_logistic_0"
    # plots.plot_hyperband_only(path, 1, 'logistic_', 'sgd_', 'mnist', 0)

    # path = "../result/tpe_logistic_0/"
    # plots.plot_tpe(path, 10, 'logistic_', 'sgd_', 'mnist', 0)
    # path = "../result/tpe_mlp_0"
    # plots.plot_tpe(path, 5, 'mlp_', 'sgd_', 'mnist', 0)

    # path1 = "../result/hyperband_mlp_0"
    # path2 = "../tpe_mlp_0"
    # paths = [path1, path2]
    # plots.plot_all(paths, 3, 5, 'mlp_', 'sgd_', 'mnist', 0)

    # path1 = "../result/hyperband_logistic_0"
    # path2 = "../tpe_logistic_0"
    # paths = [path1, path2]
    # plots.plot_all(paths, 3, 10, 'logistic_', 'sgd_', 'mnist', 0, devs=False)

    # path1 = "../result/hyperband_logistic_1"
    # path2 = "../result/tpe_logistic_1"
    # path3 = "../result/hoo_logistic_1"
    # path4 = "../result/hct_logistic_1"
    # path5 = "../result/random_logistic_1"
    # path6 = "../result/hyperloop_logistic_1"
    # paths = [path1, path2, path3, path4, path5, path6]
    # # plots.plot_hoo(path3, 1, 'logistic_', 'sgd_', 'mnist', 0)
    # plots.plot_all(paths, 10, 'logistic_', 'sgd_', 'mnist', 1, 'epochs', type_plot='linear', devs=False)

    # path1 = "../result/hyperband_cnn_1"
    # path2 = "../result/tpe_cnn_1"
    # path3 = "../result/hoo_cnn_1"
    # path4 = "../result/hct_cnn_1"
    # path5 = "../result/random_cnn_1"
    # path6 = "../result/hyperloop_cnn_1"
    # paths = [path1, path2, path3, path4, path5]
    # # plots.plot_hoo(path3, 1, 'mlp_', 'sgd_', 'mnist', 0)
    # plots.plot_all(paths, 1, 'cnn_', 'sgd_', 'mnist', 1, 'epochs', type_plot='linear', devs=False)

    # names = ['ada_', 'gbm_', 'knn_', 'rf_', 'sk_mlp_', 'svm_', 'tree_']
    # for name in names:
    #     path = "../result/hyperband_" + name + "0"
    #     plots.plot_hyperband_only(path, 1, name, '', 'wine', 0)

    # path = "../result/hct_gbm_2"
    # plots.plot_hct(path, 1, 'gbm_', '', 'breast_cancer', 2)
    # path = "../result/pct_gbm_2"
    # plots.plot_pct(path, 4, 'gbm_', '', 'breast_cancer', 2)

    # names = ['ada_', 'gbm_', 'knn_']
    exp_index = 3
    start = 0
    end = 100
    names = ['svm_']
    # names = ['mlp_']
    for name in names:
        path0 = "../result/hyperband_" + name + str(exp_index)
        path1 = "../result/tpe_" + name + str(exp_index)
        path2 = "../result/gpo_" + name + str(exp_index)
        path3 = "../result/random_" + name + str(exp_index)
        path4 = "../result/hyperloop_" + name + str(exp_index)
        path5 = "../result/dttts_" + name + str(exp_index)
        paths = [path0, path1, path2, path3, path4, path5]
        # paths = [path0, path1, path4]
        # plots.plot_all(paths, start, end, name, '', 'mnist_test', exp_index, 'iterations',
        #                plot='test', devs=False, marker=False)
        plots.plot_all(paths, start, end, name, '', 'adult', exp_index, 'iterations',
                       plot='valid', devs=True, marker=False)

    # path = "../result/hyperloop_svm_2/"
    # plots.plot_hyperloop_only(path, 1, '', 'svm_', 'breast_cancer', 2)
