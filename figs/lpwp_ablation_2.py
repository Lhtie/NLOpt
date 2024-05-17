import json
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

colors = ["#c82423", "#FF7F50", "#3DB070", "#2878b5"]
data = [
    [25.7, 33.7, 24.8, 15.8],
    [19.8, 21.8, 51.5, 6.9],
    [33.7, 46.5, 14.9, 5.0],
]
titles = [
    "w/o Reflection", "w/o Debugging", "Both"
]

# myfont = matplotlib.font_manager.FontProperties(fname=r"KaiTi_BG2312.ttf", size=20)
# matplotlib.rcParams['font.sans-serif'] = ['sans-serif']

fig = plt.figure(figsize=(14, 16))
xtick = ["Correct Answer", "Wrong Answer", "Runtime Error", "Modeling Failure"]
width = 0.2
plt.bar(np.arange(4) - width, data[0], width=width, color=colors[0], label=titles[0])
for x, y in zip(np.arange(4) - width, data[0]):
    plt.text(x, y + 0.1, str(y), ha='center', fontsize=20)  # 调整位置和字号
plt.bar(np.arange(4), data[1], width=width, color=colors[1], label=titles[1])
for x, y in zip(np.arange(4), data[1]):
    plt.text(x, y + 0.1, str(y), ha='center', fontsize=20)  # 调整位置和字号
plt.bar(np.arange(4)  + width, data[2], width=width, color=colors[2], label=titles[2])
for x, y in zip(np.arange(4) + width, data[2]):
    plt.text(x, y + 0.1, str(y), ha='center', fontsize=20)  # 调整位置和字号

plt.xticks(np.arange(4), labels=xtick, rotation=30, fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Percentage (%)", fontsize=25)
plt.legend(fontsize=20)

    
plt.tight_layout()
plt.savefig("lpwp_ablation_2.png", dpi=400)