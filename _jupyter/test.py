import numpy as np

a = np.random.random((3, 12, 12))
value = a[:, -3:]
print(value)