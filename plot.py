import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,5)

plt.plot(t, np.sin(t), label=r'$sin(t)$')
plt.plot(t, np.cos(t), label=r'$\cos(t)$')

plt.xlabel(r'$t$')

plt.tight_layout()

plt.savefig('build/plot.pdf')
