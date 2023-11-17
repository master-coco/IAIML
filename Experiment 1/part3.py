#3. Use matplotlib to plot a graph
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 20]

plt.plot(x, y, marker='s')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')

plt.show()