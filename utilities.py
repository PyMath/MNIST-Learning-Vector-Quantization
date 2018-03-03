import numpy as np
from mnist import MNIST
import scipy as sp
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    """
    Generate a mean confidence interval for performance of the prototypes

    :param data: list of accuracy values
    :type data: list
    :param confidence: confidence value
    :type confidence: float
    :return: bounds for mean compudence iterval
    """
    a = 1.0*np.asarray(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def get_mnist():
    """ Grab the mnist dataset """
    mndata = MNIST('./data/')
    train_x, train_y = mndata.load_training()
    test_x, test_y = mndata.load_testing()
    print("Loaded MNIST")
    return train_x, train_y, test_x, test_y


def get_prototypes(m, train_x, train_y, num_classes):
    """
    Gets the amount of prototypes specified from the dataset
    :param m: int, number of prototypes to select
    :type m: int
    :param train_x: training set to select prototype from
    :type train_x: numpy.ndarray
    :param train_y: training labels
    :type train_y: numpy.ndarray
    :param num_classes: how many classes in the dataset
    :type num_classes: int
    :return: prototypes selected and labels
    :rtype: numpy.ndarray, numpy.ndarray
    """
    num_proto = int(m / num_classes)
    num_col = train_x.shape[1]
    means = np.zeros((num_classes, num_col))

    for i in range(10):
        counter = 0
        total_i = np.zeros((1, num_col))
        for j in range(train_y.shape[0]):
            if train_y[j] == i:
                counter += 1
                total_i += train_x[j]
        means[i] = total_i / counter

    # Construct prototypes with 10 closest correctly classified labels to centroid
    prototypes = np.zeros((m, num_col))
    prototype_labels = np.zeros((m, 1))
    for i in range(num_classes):
        print("Adding class %d" % i)
        idx = train_y[train_y == i]
        t_x = train_x[idx, :]
        dist = (t_x - means[i]) ** 2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        sort_idx = np.argsort(dist)[:num_proto]
        prototypes[i*num_proto:i*num_proto+num_proto, :] = t_x[sort_idx]
        prototype_labels[i*num_proto:i*num_proto+num_proto] = i

    return prototypes, prototype_labels

