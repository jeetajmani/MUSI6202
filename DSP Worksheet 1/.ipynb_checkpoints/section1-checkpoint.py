# 1 Random Signals & Probabilities

## 1.1 Create Random Sequences
import numpy as np

# np.random.random is [0-1)
# multiplying by 2 makes it [0-2)
# subtracting 1 makes it [-1,1)
a = 2 * np.random.random(10000) - 1
b = 2 * np.random.random(10000) - 1

## 1.2 Plot and Discuss the Distributions
import matplotlib.pyplot as plt

# Horizontally stacked subplots showing a, b
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Probability Mass Functions of a[n] and b[n]')

# weights=np.ones_like(a)/len(a)
# shows probabilites instead of frequency
ax1.hist(a, 100, weights=np.ones_like(a)/len(a))
ax1.set_xlabel('Value of a[n]')
ax1.set_ylabel('Probability p_X(x)')

ax2.hist(b, 100, color='orange', weights=np.ones_like(b)/len(b))
ax2.set_xlabel('Value of b[n]')
ax2.set_ylabel('Probability p_X(x)')

# fig.savefig("1_2.png", format="png", bbox_inches="tight", dpi=300)
plt.show()

## 1.3 Add The Sequences
# using np.add for element-wise addition 
y = np.add(a, b)

## 1.4 Plot and Discuss the Distribution
plt.title("Probability Mass Function of y[n]")
plt.hist(y, 100, weights=np.ones_like(y)/len(y))
plt.xlabel('Value of y[n]')
plt.ylabel('Probability p_X(x)')
# plt.savefig("1_3.png", format="png", bbox_inches="tight", dpi=300)
plt.show()