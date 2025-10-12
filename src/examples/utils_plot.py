import numpy as np
import matplotlib.pyplot as plt


def plot_eval_history(eval_history, metric: str = 'log-loss'):
    """ Plot training and validation loss curves """

    n = len(eval_history['train'][metric])

    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(n),
             np.array(eval_history['train'][metric]),
             label=f'{metric} train')
    plt.plot(np.arange(n),
             np.array(eval_history['val'][metric]),
             label=f'{metric} val')
    plt.legend()

    return fig


def plot_quantiles(
        qs,
        observed_fractions_tr,
        observed_fractions_va
):
    """ Compare predicted with observed fractions """

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(qs, observed_fractions_tr, s=20, marker='x')
    plt.plot([0, 1], [0, 1], color='grey', alpha=.4, label='training data')
    plt.xlabel('predicted quantiles')
    plt.ylabel('observed fractions')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(qs, observed_fractions_va, s=20, marker='x')
    plt.plot([0, 1], [0, 1], color='grey', alpha=.4, label='validation data')
    plt.xlabel('predicted quantiles')
    plt.ylabel('observed fractions')
    plt.legend()

    return fig


def plot_means_vs_std(
        y_tr_hat_mean,
        y_tr_hat_std,
        y_va_hat_mean,
        y_va_hat_std
):
    """ Compare the predicted means with predicted STDs """

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(y_tr_hat_mean, y_tr_hat_std, s=5, alpha=.3, label='training data')
    plt.xlabel('dist mean')
    plt.ylabel('dist std')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_va_hat_mean, y_va_hat_std, s=5, alpha=.3, label='validation data')
    plt.xlabel('dist mean')
    plt.ylabel('dist std')
    plt.legend()

    return fig


def plot_true_vs_predicted_dist_means(
        dist_hat_mean_tr,
        dist_mean_tr,
        dist_hat_mean_va,
        dist_mean_va
):
    """ Compare predicted with true distribution means used in the
    data generation process (possible only for the synthetically generated data)
    """

    fig = plt.figure(figsize=(12, 5))
    kwargs = dict(alpha=.1, s=5, )

    plt.subplot(1, 2, 1)
    xy_min, xy_max = dist_mean_tr.min(), dist_mean_tr.max()

    plt.scatter(
        dist_mean_tr,
        dist_hat_mean_tr,
        label='training dataset', **kwargs)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color='grey', alpha=.3)
    plt.xlabel('dist mean')
    plt.ylabel('dist_hat mean')
    plt.legend()

    plt.subplot(1, 2, 2)
    xy_min, xy_max = dist_mean_va.min(), dist_mean_va.max()

    plt.scatter(
        dist_mean_va,
        dist_hat_mean_va,
        label='test dataset', **kwargs)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color='grey', alpha=.3)
    plt.xlabel('dist mean')
    plt.ylabel('dist_hat mean')
    plt.legend()

    plt.show()

    return fig
