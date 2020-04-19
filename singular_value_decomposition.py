import numpy as np
from numpy import linalg as LA

m, n = 2, 3
A = np.random.rand(m, n)
U, S, V = LA.svd(A)

print(A)
print(U)
print(S)
print(V)