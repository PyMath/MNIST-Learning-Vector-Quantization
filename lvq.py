import numpy as np
import argparse
import utilities as utils

parser = argparse.ArgumentParser(description="Specify hyperparameters for scripts")
parser.add_argument("-np", "--num_protos", type=int, default=10000, help="Specify number of prototypes to select")
parser.add_argument("-nc", "--num_classes", type=int, default=10,
                    help="Specify number of classes, default 10 assumes MNIST")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.3, help="Learning rate to train prototypes")
parser.add_argument("-e", "--num_epochs", type=int, default=5, help="Number of epochs to train prototypes")
parser.add_argument("-nt", "--num_train", type=int, default=3, help="Number of times to perform training")
parser.add_argument("-mc", "--mean_confidence", type=float, default=0.95, help="degree for mean confidence interval")
parser.add_argument("-save", type=str, default='./saves/protos', help="Where to save best performing prototypes")

args = parser.parse_args()


def get_best_matching_unit(protos_x, row):
    """
    Grab the prototype that is closest (euclidean distance) to the row
    :param protos_x: list of prototypes
    :type protos_x: list
    :param row: current row inside MNIST data
    :type row: numpy.ndarray
    :return:prototype that is closest distance and the index associated with it
    :rtype: numpy.ndarray, numpy.int64
    """
    dist = (protos_x - row) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    min_idx = np.argmin(dist)
    return protos_x[min_idx], min_idx


def train_prototypes(protos_x, protos_y, train_x, train_y, num_epochs, lr):
    """
    Train the prototypes so that that can be representative of the training set

    :param protos_x: prototypes to train
    :type protos_x: numpy.ndarray
    :param protos_y: Prototype labels associated with each one
    :type protos_y: numpy.ndarray
    :param train_x: training points to improve prototypes
    :type train_x: numpy.ndarray
    :param train_y: labels associated with prototypes
    :type train_y: numpy.ndarray
    :param num_epochs: number of epochs to train for
    :type num_epochs: int
    :param lr: learning rate for training
    :type lr: float
    :return: trained prototypes and their associated labels
    :rtype: numpy.ndarray, numpy.ndarray
    """
    for epoch in range(1, num_epochs+1):
        rate = lr * (1.0 - (epoch / num_epochs))

        sum_error = 0.0
        for i, row in enumerate(train_x):
            best_match, idx = get_best_matching_unit(protos_x, row)
            if i % 1500 == 0:
                print("Iteration: %d" % i)
            error = train_x[i] - protos_x[idx]
            if protos_y[idx] == train_y[i]:
                protos_x[idx] += rate * error
            else:
                protos_x[idx] -= rate * error
            sum_error += np.sum(error ** 2)
        print("Epoch: " + str(epoch) + "\tLR: " + str(rate) + "\tSSE: " + str(sum_error/len(sum_error)))
    return protos_x, protos_y


def predict(test_x, test_y, protos_x, protos_y):
    """
    Use the trained prototypes to test their performance against MNIST test set

    :param test_x: input for the test set
    :type test_x: numpy.ndarray
    :param test_y: labels for the test set
    :type test_y: numpy.ndarray
    :param protos_x: prototypes that were trained
    :type protos_x: numpy.ndarray
    :param protos_y: labels for the prototypes
    :type protos_y: numpy.ndarray
    :return: accuracy of the training set
    :rtype: numpy.float64
    """
    pred_y = np.zeros(test_y.shape)
    for i in range(test_x.shape[0]):
        if i % 1000 == 0:
            print(i)
        test = test_x[i]
        dist = np.square(protos_x - test)
        dist = np.sum(dist,axis=1)
        dist = np.sqrt(dist)
        min_idx = np.argmin(dist)
        pred_y[i] = protos_y[min_idx]
    acc = np.sum(pred_y == test_y)/test_y.shape[0]
    return acc


def main():
    train_x, train_y, val_x, val_y = utils.get_mnist()

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    val_x = np.asarray(val_x)
    val_y = np.asarray(val_y)
    accuracies = []
    protos_x_final = np.asarray((args.num_train, args.num_protos, train_x.shape[1]))
    protos_y_final = np.asarray((args.num_train, args.num_protos, 1))
    for i in range(args.num_train):
        protos_x, protos_y = utils.get_prototypes(args.num_protos, train_x, train_y, args.num_classes)
        protos_x, protos_y = train_prototypes(protos_x, protos_y, train_x, train_y, args.num_epochs, args.learning_rate)
        protos_x_final[i] = protos_x
        protos_y_final[i] = protos_y
        acc = predict(val_x, val_y, protos_x, protos_y)
        accuracies.append(acc)
    utils.mean_confidence_interval(accuracies, confidence=args.mean_confidence)
    highest_acc = max(accuracies)
    highest_idx = 0
    for idx, val in enumerate(accuracies):
        if highest_acc == val:
            highest_idx = idx
            break
    print("Prototype set %d performed the best" % highest_idx)
    np.save(args.save + '-x', protos_x_final[highest_idx, :, :])
    np.save(args.save + '-y', protos_y_final[highest_idx, :, :])


if __name__ == "__main__":
    main()
