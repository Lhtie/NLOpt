import json
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

colors = ["#c82423", "#FF7F50", "#3DB070", "#2878b5"]
data = [
    [42.9, 35.7, 14.3, 7.1],
    [36.4, 36.4, 27.3, 0.0],
    [10.0, 90.0, 0.0, 0.0],
    [0.0, 54.5, 45.5, 0.0]
]
titles = [
    "Introduction to linear optimization",
    "Lectures in lp modeling",
    "Linear and convex optimization",
    "Model building in math programming"
]

# myfont = matplotlib.font_manager.FontProperties(fname=r"KaiTi_BG2312.ttf", size=20)
# matplotlib.rcParams['font.sans-serif'] = ['sans-serif']

fig = plt.figure(figsize=(14, 16))
xtick = ["Correct Answer", "Wrong Answer", "Runtime Error", "Modeling Failure"]
width = 0.2
plt.bar(np.arange(4) - width * 2 + width / 2, data[0], width=width, color=colors[0], label=titles[0])
for x, y in zip(np.arange(4) - width * 2 + width / 2, data[0]):
    plt.text(x, y + 0.1, str(y), ha='center', fontsize=20)  # 调整位置和字号
plt.bar(np.arange(4) - width  + width / 2, data[1], width=width, color=colors[1], label=titles[1])
for x, y in zip(np.arange(4) - width + width / 2, data[1]):
    plt.text(x, y + 0.1, str(y), ha='center', fontsize=20)  # 调整位置和字号
plt.bar(np.arange(4)  + width / 2, data[2], width=width, color=colors[2], label=titles[2])
for x, y in zip(np.arange(4) + width / 2, data[2]):
    plt.text(x, y + 0.1, str(y), ha='center', fontsize=20)  # 调整位置和字号
plt.bar(np.arange(4) + width  + width / 2, data[3], width=width, color=colors[3], label=titles[3])
for x, y in zip(np.arange(4) + width + width / 2, data[3]):
    plt.text(x, y + 0.1, str(y), ha='center', fontsize=20)  # 调整位置和字号

plt.xticks(np.arange(4), labels=xtick, rotation=30, fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Percentage (%)", fontsize=25)
plt.legend(fontsize=20)

    
plt.tight_layout()
plt.savefig("nlp4lp_brench.png", dpi=400)