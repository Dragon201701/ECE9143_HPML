import matplotlib.pyplot as plt
import numpy as np


x1 = np.linspace(0, 6.67, 50)
y1 = x1 * 30

x2 = np.linspace(6.67, 10, 50)
y2 = x2 - x2 + 200

x3 = np.linspace(6.67, 6.67, 50)
y3 = np.linspace(0, 200, 50)

at1 = 1000000 * 8 / 1073741824
at2 = 300000000 * 8 / 1073741824

c1x1 = at1
c1y1 = 6.461 * at1

c1x2 = at2
c1y2 = 6.012 * at2

c2x1 = at1
c2y1 = 20.072 * at1

c2x2 = at2
c2y2 = 10.925 * at2

c3x1 = at1
c3y1 = 23.468 * at1

c3x2 = at2
c3y2 = 12.610 * at2

c4x1 = at1
c4y1 = 0.026 * at1

c4x2 = at2
c4y2 = 0.026 * at2

c5x1 = at1
c5y1 = 22.417 * at1

c5x2 = at2
c5y2 = 13.202 * at2

plt.figure()
plt.xlabel("Arith. Intensity [FLOP/Bytes]")
plt.ylabel("Actual FLOPS")
plt.plot(x1, y1, color = 'blue', label = '30 GB/S')
plt.plot(x2, y2, color = 'red', label = '200 GFLOPS')
plt.plot(x3, y3, color = 'green', linestyle = '--', label = 'X = 6.667')
plt.scatter(c1x1, c1y1, label = 'dp1 small', color = 'red')
plt.scatter(c2x1, c2y1, label = 'dp2 small', color = 'red')
plt.scatter(c3x1, c3y1, label = 'dp3 small', color = 'red')
plt.scatter(c4x1, c4y1, label = 'dp4 small', color = 'red')
plt.scatter(c5x1, c5y1, label = 'dp5 small', color = 'red')
plt.scatter(c1x2, c1y2, label = 'dp1 large', color = 'blue')
plt.scatter(c2x2, c2y2, label = 'dp2 large', color = 'blue')
plt.scatter(c3x2, c3y2, label = 'dp3 large', color = 'blue')
plt.scatter(c4x2, c4y2, label = 'dp4 large', color = 'blue')
plt.scatter(c5x2, c5y2, label = 'dp5 large', color = 'blue')
plt.legend(loc = 'upper left')
plt.show()