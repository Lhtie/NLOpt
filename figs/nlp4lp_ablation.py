import json
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

colors = ["#c82423", "#FF7F50", "#3DB070", "#2878b5"]
data = [
    [17.4, 43.5, 28.3, 10.9],
    [19.6, 54.4, 23.9, 2.2],
    [23.9, 52.2, 21.7, 2.2]
]
titles = ["Correct Answer", "Wrong Answer", "Runtime Error", "Modeling Failure"]

# myfont = matplotlib.font_manager.FontProperties(fname=r"KaiTi_BG2312.ttf", size=20)
# matplotlib.rcParams['font.sans-serif'] = ['sans-serif']

fig = plt.figure(figsize=(16, 18))
xtick = ["Frontend(10) Backend(15)", "Frontend(20) Backend(20)", "Frontend(40) Backend(40)"]
for i in range(4):
    plt.plot(np.arange(3), [x[i] for x in data], label=titles[i], color=colors[i], marker=".", markersize=50, linewidth=10)
    for x, y in zip(np.arange(3), [x[i] for x in data]):
        plt.text(x, y + 1, str(y), ha='center', fontsize=25)  # 调整位置和字号

plt.xticks(np.arange(3), labels=xtick, rotation=30, fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Percentage (%)", fontsize=25)
plt.ylim(0, np.max(data) + 15)
plt.legend(fontsize=20)

    
plt.tight_layout()
plt.savefig("nlp4lp_ablation.png", dpi=400)