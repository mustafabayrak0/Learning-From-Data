import numpy as np


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class minMaxScaler():
    def __init__(self, x, min_limit=0, max_limit=1):
        self.x = x
        self.min = np.min(x, axis=0)
        self.max = np.max(x, axis=0)
        self.min_limit = min_limit
        self.max_limit = max_limit

    def transform(self, x):
        ##############################################################################
        # TODO: Implement min max scaler function that scale a data range into       #
        # min_limit and max_limit                                                    #
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.   #
        # MinMaxScaler.html                                                          #
        ##############################################################################
        # Replace "pass" statement with your code

        # Scale the data to the specified range
        x_std = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
        x_scaled = x_std * (self.max_limit - self.min_limit) + self.min_limit
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return x_scaled

    def inverse_transform(self, x_scaled):
        ##############################################################################
        # TODO: Implement inverse min max scaler that scale a data range back into   #
        # previous range.                                                            #
        ##############################################################################
        # Replace "pass" statement with your code

        # unscales the given x with the parameters
        x = ((x_scaled - self.min_limit) / (self.max_limit - self.min_limit)) * (self.max - self.min) + self.min
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return x


def leastSquares(x, y):
    ##############################################################################
    # TODO: Implement least square theorem.                                      #
    #                                                                            #
    # You may not use any built in function which directly calculate             #
    # Least squares except matrix operation in numpy.                            #
    ##############################################################################
    # Replace "pass" statement with your code
    x = np.insert(x, 0, 1, axis=1)
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return w


class gradientDescent():
    def __init__(self, x, y, w, lr, num_iters):
        self.x = np.c_[np.ones((x.shape[0], 1)), x]
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w
        self.weight_history = [self.w]
        self.cost_history = [np.sum(np.square(self.predict(self.x) - self.y)) / self.x.shape[0]]

    def gradient(self):
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code
        l = self.x.shape[0]
        gradient = (1 / l) * np.dot(np.transpose(self.x), np.dot(self.x, self.w) - self.y)
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return gradient

    def fit(self, lr=None, n_iterations=None):
        k = 0
        # Check if diminishing
        decrease = False
        if n_iterations is None:
            n_iterations = self.num_iters
        if lr is not None:
            if lr != "diminishing":
                self.lr = lr
            else:
                decrease = True
                initial_lr = self.lr
        ##############################################################################
        # TODO: Implement gradient descent algorithm.                                #
        #                                                                            #
        # You may not use any built in function which directly calculate             #
        # gradient.                                                                  #
        # Steps of gradient descent algorithm:                                       #
        #   1. Calculate gradient of cost function. (Call gradient() function)       #
        #   2. Update w with gradient.                                               #
        #   3. Log weight and cost for plotting in weight_history and cost_history.  #
        #   4. Repeat 1-3 until the cost change converges to epsilon or achieves     #
        # n_iterations.                                                              #
        # !!WARNING-1: Use Measn Square Error between predicted and  actual y values #
        # for cost calculation.                                                      #
        #                                                                            #
        # !!WARNING-2: Be careful about lr can takes "diminishing". It will be used  #
        # for further test in the jupyter-notebook. If it takes lr decrease with     #
        # iteration as (lr =(1/k+1), here k is iteration).                           #
        ##############################################################################
        while k < n_iterations:
            if decrease:
                self.lr = 1 / (k + 1)
            self.w = self.w - (self.lr * self.gradient())
            # Calculate cost
            cost = np.mean(np.square(self.y - self.predict(self.x)))
            # Add to weight history
            self.weight_history.append(self.w)
            # Add to cost history
            self.cost_history.append(cost)
            if abs(cost - self.cost_history[-2]) < self.epsilon:
                break
            k += 1
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return self.w, k

    def predict(self, x):
        y_pred = np.zeros_like(self.y)
        y_pred = x.dot(self.w)
        return y_pred
