import numpy as np
from matplotlib import pyplot as plt

a = np.load('iter_900_cmap_h.npy')
b = np.load('iter_900_cmap_v.npy')
print(a.shape)
exit(123)
c = (a + b)[:, :, 0]
c = c.flatten()

print(np.mean(c))
plt.figure(figsize=(8, 8))
plt.scatter(list(range(len(c))), c, s=0.2)
plt.savefig('test.png')

