import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("boosting_error.csv", delimiter=',', 
                     names=['x', 'y'])

plt.plot(data['x'], data['y'], color='b')
plt.show()