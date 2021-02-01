import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(19680801)

mu = 200
sigma = 25
n_bins = 50
x = np.random.normal(mu, sigma, size=100)

fname = "./pixel_compression/res_compress_rate_KMNIST_d1_1_d0_1.csv"

old = pd.read_csv(fname, header=None, usecols = [1])
new = pd.read_csv(fname, header=None, usecols = [2])
ratio = np.divide(new, old)

print(ratio)

fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
n, bins, patches = ax.hist(ratio, n_bins, density=True, histtype='step',
                           cumulative=True, label='Compression Ratio')

# Add a line showing the expected distribution.
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
# y = y.cumsum()
# y /= y[-1]
#
# ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

# Overlay a reversed cumulative histogram.
# ax.hist(x, bins=bins, density=True, histtype='step', cumulative=-1,
#         label='Reversed emp.')

# tidy up the figure
# ax.grid(True)
ax.legend(loc='right')
ax.set_title('Cumulative compression ratio through topologization')
ax.set_xlabel('Ratio')
ax.set_ylabel('Cumulative probability')

plt.show()