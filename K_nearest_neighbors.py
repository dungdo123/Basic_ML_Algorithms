# Choose value for K
# Randomly select K featuresets to start as your centroids
# Calculate distance of all other featuresets to centroids
# Classify other featuresets as same as closest centroid
# Take mean of each class (mean of all featuresets by class), making that mean the new centroid
# Repeat steps 3-5 until optimized (centroids no longer moving)

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()
class K_means:
    def __init__(self,k=2,tol=0.001,max_iter = 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
        for i in range(self.max_iter):
            self.classification = {}

            for i in range(self.k):
                self.classification[i] = []
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            rev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)









