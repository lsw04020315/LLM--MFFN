import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

setting = 'MFFN-QBO-96-96_sl96_ll48_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0'

pred = np.load('./results/'+setting+'/pred.npy')
true = np.load('./results/'+setting+'/true.npy')

plt.figure(figsize=(12, 6))  # 调整画布尺寸

# 绘制真实值和预测值（加粗线条）
plt.plot(true[-1, :, -1],  label='GroundTruth',linewidth=2.5, linestyle='-', color='blue')
plt.plot(pred[-1, :, -1],  label='Prediction', linewidth=2.5, linestyle='-', color='red')

# 设置图例（加粗且增大字号）
plt.legend(prop={'size': 17, 'weight': 'bold'})

# 坐标轴边框加粗
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(axis='both', which='major', width=2)
# 设置坐标轴刻度字体（加粗且增大字号）
plt.xticks(fontsize=22, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')

# 添加网格线（可选）
plt.grid(alpha=0.3, linestyle='--')

plt.show()
