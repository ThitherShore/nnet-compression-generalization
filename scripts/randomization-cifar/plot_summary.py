accs = [[1.0, 1.0, 1.0, 1.0, 0.97896], 
		[1.0, 1.0, 0.97798, 0.69102, 0.62264], 
		[1.0, 0.99968, 0.92822, 0.60466, 0.4194], 
		[1.0, 0.9952, 0.87842, 0.64598, 0.27864], 
		[0.99962, 0.98822, 0.8876, 0.67422, 0.2077]]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(8, 4.5))

colors = ["steelblue", "orange", "yellowgreen", "tomato", "slateblue"]
levels = ["0%", "30%", "50%", "70%", "100%"]  # level of label randomization
sparsity = [50, 60, 70, 80, 90]
for i, accs_per_level in enumerate(accs):
	plt.plot(sparsity, accs_per_level, 
		marker='', color=colors[i], linewidth=2, alpha=0.9, label=levels[i])

plt.gca().set_xlim(left=50)
plt.gca().set_ylim(bottom=0)
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.xlabel("sparsity")
plt.ylabel("accuracy (train)")
plt.xticks([50, 60, 70, 80, 90])
# plt.legend(loc="lower left")
for y, label in zip(np.array(accs)[:, -1], levels):
	plt.text(91, y, label)
plt.savefig("plot_summary.png", dpi=300)
