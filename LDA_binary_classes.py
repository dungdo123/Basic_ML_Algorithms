from __future__ import division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
np.random.seed(22)

means = [[0, 5],[5, 0]]
cov0 = [[4,3],[3,4]]
cov1 = [[3,1],[1,1]]

N0 = 50
N1 = 40
N = N0 + N1

X0 = np.random.multivariate_normal(means[0], cov0, N0)
X1 = np.random.multivariate_normal(means[0], cov1, N1)

m0 = np.mean(X0.T, axis=1, keepdims=True)
m1 = np.mean(X1.T, axis=1, keepdims=True)

a = (m0 - m1)
S_B = a.dot(a.T)

A = X0.T - np.tile(m0, (1, N0))
B = X1.T - np.tile(m1, (1, N1))

S_W = A.dot(A.T) + B.dot(B.T)

_, W = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
w = W[:,0]
print(w)

