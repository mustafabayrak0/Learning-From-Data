import numpy as np


# Do not import any other libraries.

# Mustafa Bayrak 150210339
class KMeans:

    def __init__(self, X, n_clusters):
        self.X = X
        self.n_clusters = n_clusters
        self.centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    def mean(self, value):
        """
        Calculate mean of the dataset column-wise.
        Do not use built-in functions

        :param value: data
        :return the mean value
        """

        # Initialize sum to 0 for each column
        total = [0 for _ in range(len(value[0]))]

        # Loop through each data point
        for v in value:
            # Add points to their columns
            for i in range(len(v)):
                total[i] += v[i]

        # Find means column-wise
        mean = [t / len(value) for t in total]

        return mean

    def std(self):
        """
        Calculate standard deviation of the dataset.
        Use the mean function you wrote. Do not use built-in functions

        :param X: dataset
        :return the standard deviation value
        """

        # Calculate means
        mean_values = self.mean(self.X)

        # Initialize sum of squared deviations to 0
        ssd = [0 for i in range(len(self.X[0]))]

        # Loop through each data point
        for x in self.X:
            # Add the squared deviation to the sum
            for i in range(len(x)):
                ssd[i] += (x[i] - mean_values[i]) ** 2

        # Divide the ssd by the number of data points and take the square root to get the standard deviation
        std_dev = [(s / len(self.X)) ** 0.5 for s in ssd]


        return std_dev

    def standard_scaler(self):
        """
        Implement Standard Scaler to X.
        Use the mean and std functions you wrote. Do not use built-in functions

        :param X: dataset
        :return X_scaled: standard scaled X
        """
        # Calculate means
        mean_values = self.mean(self.X)
        # Calculate standard deviations
        std_values = self.std()
        # Return scaled X
        return (self.X - mean_values) / std_values

    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two data points.
        Do not use any external libraries

        :param point1: data point 1, list of floats
        :param point2: data point 2, list of floats

        :return the Euclidean distance between two data points
        """
        # Return euclidean distance
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    def get_closest_centroid(self, point):
        """
        Find the closest centroid given a data point.

        :param point: list of floats
        :param centroids: a list of list where each row represents the point of each centroid
        :return: the number(index) of the closest centroid
        """
        # Find distances to centroids
        distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
        # Return the index of the closest centroid
        return np.argmin(distances)

    def update_clusters(self):
        """
        Assign all data points to the closest centroids.
        Use "get_closest_centroid" function

        :return: cluster_dict: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid
        Example:
        list_of_points = [[1.1, 1], [4, 2], [0, 0]]
        centroids = [[1, 1],
                    [2, 2]]

            print(update_clusters())
        Output:
            {'0': [[1.1, 1], [0, 0]],
             '1': [[4, 2]]}
        """
        clusters = {i: [] for i in range(self.n_clusters)}
        # Find the closest centroid for each point and append that point to cluster of the closest centroid
        for point in self.X:
            closest_centroid = self.get_closest_centroid(point)
            clusters[closest_centroid].append(point)
        return clusters

    def update_centroids(self, cluster_dict):
        """
        Update centroids using the mean of the given points in the cluster.
        Doesn't return anything, only change self.centroids
        Use your mean function.
        Consider the case when one cluster doesn't have any point in it !

        :param cluster_dict: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid

        """
        for i in range(self.n_clusters):
            if cluster_dict[i]:  # if the cluster is not empty
                self.centroids[i] = self.mean(np.array(cluster_dict[i]))
            else:  # if the cluster is empty
                self.centroids[i] = self.X[np.random.choice(len(self.X))]  # reinitialize to a random point

    def converged(self, clusters, old_clusters):
        """
        Check the clusters converged or not

        :param clusters: new clusters, dictionary where keys are cluster labels and values are the points(list of list)
        :param old_clusters: old clusters, dictionary where keys are cluster labels and values are the points(list of list)
        :return: boolean value: True if clusters don't change
        Example:
        clusters = {'0': [[1.1, 1], [0, 0]],
                    '1': [[4, 2]]}
        old_clusters = {'0': [[1.1, 1], [0, 0]],
                        '1': [[4, 2]]}
            print(update_assignment(clusters, old_clusters))
        Output:
            True
        """
        # Check if cluster keys are equal or not
        for key in clusters.keys():
            if not np.array_equal(clusters[key], old_clusters[key]):
                return False
        return True

    def calculate_wcss(self, clusters):
        """

        :param clusters: dictionary where keys are clusters labels and values the data points belong to that cluster
        :return:
        """
        wcss = 0
        # For each cluster, find the sum of squared euclidean distances
        for i in range(self.n_clusters):
            for point in clusters[i]:
                wcss += self.euclidean_distance(point, self.centroids[i]) ** 2
        return wcss

    def fit(self):
        """
        Implement K-Means clustering until the clusters don't change.
        Use the functions you have already implemented.
        Print how many steps does it take to converge.
        :return: final_clusters: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid
                 final_centroids: list of list with shape (n_cluster, X.shape[1])
                 wcss: within-cluster sum of squares
        """
        # Scale the data
        self.X = self.standard_scaler()
        old_clusters = self.update_clusters()
        # Initialize steps number
        steps = 0

        while True:
            # Update centroids
            self.update_centroids(old_clusters)
            # Update clusters
            new_clusters = self.update_clusters()
            # Increase steps count
            steps += 1
            # If clusters are converged, then break the loop
            if self.converged(new_clusters, old_clusters):
                break
            old_clusters = new_clusters
        print(f'Converged in {steps} steps')
        # Calculate WCSS
        wcss = self.calculate_wcss(new_clusters)
        return new_clusters, self.centroids.tolist(), wcss
