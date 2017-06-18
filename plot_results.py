import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("bmh")

fig, ax = plt.subplots(1, 1)

data_frame = pd.DataFrame()
data_frame["Accuracy"] = pd.read_csv("boosting_accuracy.csv", header=0, 
                                      index_col=0)
data_frame["Error"] = pd.read_csv("boosting_error.csv", header=0, index_col=0)

ax.set_xlabel("Iterations")
ax.set_ylabel("Percentage")

data_frame.plot(ax=ax)  # COMBINED
#data_frame.plot(ax=ax, legend=False)  # ACCURACY ONLY
#data_frame.plot(ax=ax, color="#A60628", legend=False)  # ERROR ONLY
plt.suptitle("AdaBoost with 5-fold Cross-validation")

plt.show()