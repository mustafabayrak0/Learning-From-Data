############################################################################################
#               Implementation of MultiClass Logistic Regression.                          #
############################################################################################


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time


def sigmoid(z):
    #     """
    #     Computes the sigmoid function element wise over a numpy array.
    #     :param z: A numpy array.
    #     :return: A numpy array of the same size of z

    ##############################################################################
    sigmoid_val = 1 / (1 + np.exp(-z))
    ##############################################################################
    return sigmoid_val


def log_loss(X, Y, W, N):
    """
    Computes the log-loss function, and its gradient over a mini-batch of data.
    :param X: The feature matrix of size (N, F+1), where F is the number of features.
    :param Y: The label vector of size (N, 1).
    :param W: The weight vector of size (F+1, 1).
    :param N: The number of samples in the mini-batch.
    :return: Loss (a scalar) and gradient (a vector of size (F+1, 1)).
    """
    ##############################################################################
    #                             YOUR CODE                                      #
    ##############################################################################
    Y_hat = sigmoid(X @ W)
    # Compute loss
    loss = np.array([-1 / N * sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))])
    # Compute gradient
    grad = 1 / N * X.T @ (Y_hat - Y)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################

    return loss[0][0], grad


def visualizer(loss, accuracy, n_epochs):
    """
    Returns the plot of Training/Validation Loss and Accuracy.
    :param loss: A list defaultdict with 2 keys "train" and "val".
    :param accuracy: A list defaultdict with 2 keys "train" and "val".
    :param n_epochs: Number of Epochs during training.
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    x = np.arange(0, n_epochs, 1)
    axs[0].plot(x, loss['train'], 'b')
    axs[0].plot(x, loss['val'], 'r')
    axs[1].plot(x, accuracy['train'], 'b')
    axs[1].plot(x, accuracy['val'], 'r')

    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss value")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy value (in %)")

    axs[0].legend(['Training loss', 'Validation loss'])
    axs[1].legend(['Training accuracy', 'Validation accuracy'])


class OneVsAll:
    def __init__(self, x_train, y_train, x_test, y_test, alpha, beta, mb, n_class, F, n_epochs, info):
        """
        This is an implementation from scratch of Multi Class Logistic Regression using One vs All strategy,
        and Momentum with SGD optimizer.

        :param x_train: Vectorized training data.
        :param y_train: Label training vector.
        :param x_test: Vectorized testing data.
        :param y_test: Label test vector.
        :param alpha: The learning rate.
        :param beta: Momentum parameter.
        :param mb: Mini-batch size.
        :param n_class: Number of classes.
        :param F: Number of features.
        :param n_epochs: Number of Epochs.
        :param info: 1 to show training loss & accuracy over epochs, 0 otherwise.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alpha = alpha
        self.beta = beta
        self.mb = mb
        self.n_class = n_class
        self.F = F
        self.n_epochs = n_epochs
        self.info = info

    def relabel(self, label):
        """
        This function takes a class, and relabels the training label vector into a binary class,
        it's used to apply One vs All strategy.

        :param label: The class to relabel.
        :return: A new binary label vector.
        """
        y = self.y_train.tolist()
        n = len(y)
        y_new = [1 if y[i] == label else 0 for i in range(n)]

        return np.array(y_new).reshape(-1, 1)

    def momentum(self, y_relab):
        """
        This function is an implementation of the momentum with SGD optimization algorithm, and it's
        used to find the optimal weight vector of the logistic regression algorithm.
        :param y_relab: A binary label vector.
        :return: A weight vector, and history of loss/accuracy over epochs.
        """

        # Initialize weights and velocity vectors
        W = np.zeros((self.F + 1, 1))
        V = np.zeros((self.F + 1, 1))

        # Store loss & accuracy values for plotting
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        # Split into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, y_relab, test_size=0.1, random_state=42)
        n_train = len(x_train)
        n_val = len(x_val)

        ##############################################################################
        #                             YOUR CODE                                      #
        ##############################################################################

        for _ in range(self.n_epochs):
            start = time.time()
            train_loss = 0
            # Compute the loss and gradient over mini-batches of the training set
            for i in range(0, n_train - self.mb + 1):
                # Create mini-batch x
                mini_batch_x = x_train[i:i + self.mb]
                # Create mini-batch y
                mini_batch_y = y_train[i:i + self.mb]
                # Find loss and gradient
                train_loss, grad = log_loss(mini_batch_x, mini_batch_y, W, self.mb)
                # Change V
                V = self.beta * V + self.alpha * grad
                # Change W
                W = W - V

            # Find probabilities
            train_proba = sigmoid(x_train @ W)
            # Classify according to probabilities
            train_class = (train_proba >= 0.5).astype(int).reshape(-1, 1)
            # Number of correct predictions
            correct_predictions_train = np.sum(train_class == y_train)
            # Training accuracy
            train_acc = 100 * correct_predictions_train / n_train
            # Compute the loss & accuracy over the validation set
            y_hat = sigmoid(x_val @ W)
            # Find loss
            val_loss, _ = log_loss(x_val, y_val, W, n_val)
            # Classify according to probabilities
            val_class = (y_hat >= 0.5).astype(int).reshape(-1, 1)
            # Number of correct predictions
            correct_predictions_val = np.sum(val_class == y_val)
            # Validation accuracy
            val_acc = 100 * correct_predictions_val / n_val

            ##############################################################################
            #                             END OF YOUR CODE                               #
            ##############################################################################

            end = time.time()
            duration = round(end - start, 2)

            if self.info: print("Epoch: {} | Duration: {}s | Train loss: {} |"
                                " Train accuracy: {}% | Validation loss: {} | "
                                "Validation accuracy: {}%".format(_, duration,
                                                                  round(train_loss, 5), train_acc, round(val_loss, 5),
                                                                  val_acc))

            # Append training & validation accuracy and loss values to a list for plotting
            loss['train'].append(train_loss)
            loss['val'].append(val_loss)
            accuracy['train'].append(train_acc)
            accuracy['val'].append(val_acc)

        return W, loss, accuracy

    def train(self):
        """
        This function trains the model using One-vs-All strategy, and returns a weight
        matrix, to be used during inference.
        :return: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        """

        weights = []
        loss, accuracy = 0, 0

        ##############################################################################
        #                             YOUR CODE                                      #
        ##############################################################################
        # Replace "pass" statement with your code

        for i in range(1, self.n_class + 1):
            print("-" * 50 + " Processing class {} ".format(i) + "-" * 50 + "\n")
            # Apply relabeling
            y_relab = self.relabel(i)
            # Find W, loss, and accuracy
            W, loss, accuracy = self.momentum(y_relab)
            # Append W to weights
            weights.append(W)

        # Get the weights matrix as a numpy array
        weights = np.array(weights)

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return weights, loss, accuracy

    def test(self, weights):
        """
        This function is used to test the model over new testing data samples, using
        the weights matrix obtained after training.
        :param weights: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        :return:
        """

        ##############################################################################
        #                             YOUR CODE                                      #
        ##############################################################################

        # Getting the probabilities matrix for each class over the testing set (ntest, n_class)
        proba = sigmoid(self.x_test @ weights)
        y_hat = [-1 for i in range(len(self.x_test))]
        probs = [0 for i in range(len(self.x_test))]
        # Classify according to probabilities
        for i in range(len(proba)):
            for j in range(len(proba[i])):
                if proba[i][j][0] >= probs[j]:
                    probs[j] = proba[i][j][0]
                    y_hat[j] = i + 1

        # Number of correct predictions
        correct_predictions = np.sum(y_hat == self.y_test)
        # Computing the test accuracy
        test_acc = 100 * correct_predictions / len(self.y_test)
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        print("-" * 50 + "\n Test accuracy is {}%".format(test_acc))

