import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

def make_colors(image, mu):
    return np.argmin(np.linalg.norm(image[:, :, :, None] - mu.T[None, None, :, :], axis=2), axis=2)

path_to_small_image = '../data/peppers-small.tiff'
path_to_large_image = '../data/peppers-large.tiff'

A = imread(path_to_small_image).astype(float)
k = 16

mu_index_ = [None, None]
for i in range(2):
    mu_index_[i] = np.random.choice(A.shape[i], k, replace=False)
mu_index = list(zip(*mu_index_))
mu = k * [None]
for i, index in enumerate(mu_index):
    mu[i] = A[index].copy()
mu = np.array(mu)

cnt = 0
eps, prev_loss, loss = 1e-3, 0, 1
while(np.abs(loss - prev_loss) > eps and cnt < 300):
    prev_loss = loss
    colors = make_colors(A, mu)
    loss = np.linalg.norm(A - mu[colors], axis=2).sum()
    print("Iteration = ", cnt, "objective = ", loss)
    for i in range(k):
        index = colors == i
        mu[i] = A[index].mean(axis=0)
    cnt += 1

A = imread(path_to_large_image).astype(float)
A_clustered = np.zeros(A.shape)
colors = make_colors(A, mu)
for i in range(k):
    index = colors == i
    A_clustered[index] = mu[i]

plt.imshow(A.astype(int))
plt.show()

plt.imshow(A_clustered.astype(int))
plt.show()
