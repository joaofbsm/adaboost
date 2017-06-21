import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("bmh")

fig, ax = plt.subplots(1, 1)

data_frame = pd.DataFrame()
#data_frame["Accuracy"] = pd.read_csv("boosting_accuracy.csv", header=0, 
#                                      index_col=0)
data_frame["5-fold Weak Model Error"] = pd.read_csv("model_error.csv", 
                                                    header=0, index_col = 0)
data_frame["5-fold Ensemble Error"] = pd.read_csv("boosting_error.csv", 
                                                  header=0, index_col=0)

#data_frame["Weak Model Error"] = pd.read_csv("model_error_nok.csv", header=0, 
#                                             index_col = 0)
#data_frame["Ensemble Error"] = pd.read_csv("boosting_error_nok.csv", header=0, 
#                                           index_col=0)

ax.set_xlabel("Iterations")
ax.set_ylabel("Percentage")

data_frame.plot(ax=ax, color=[ "#D55E00", "#A60628"])
#data_frame.plot(ax=ax, color=[ "#D55E00", "#A60628", "#56B4E9", "#0072B2"], 
#                linewidth=1)  # COMPLETE VS K-FOLD
#data_frame.plot(ax=ax, legend=False)  # ACCURACY ONLY
#data_frame.plot(ax=ax, color=["#D55E00", "#A60628"])  # ERROR ONLY
#plt.suptitle("Entire dataset vs. 5-fold cross-validation")
plt.suptitle("AdaBoost with 5-fold cross-validation")
plt.show()