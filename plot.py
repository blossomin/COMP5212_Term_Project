import matplotlib.pyplot as plt
import sys
import os
import numpy as np
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

### result 1
file_eff = "./eff_models_v3/log.txt"
file_res = "./res_models_v3/log.txt"


rfile_eff = open(file_eff, "r")
rfile_res = open(file_res, "r")


rlines_eff = rfile_eff.readlines()
rlines_res = rfile_res.readlines()

train_loss_eff, val_loss_eff, val_accu_eff, val_f1_eff = [],[],[],[]
train_loss_res, val_loss_res, val_accu_res, val_f1_res = [], [] ,[], []


print(f"len:{len(rlines_eff)}")
for i in range(len(rlines_eff) - 1):
    train_loss_eff.append(float(rlines_eff[i].split(",")[2].split(":")[1]))
    val_loss_eff.append(float(rlines_eff[i].split(",")[3].split(":")[1]))
    val_accu_eff.append(float(rlines_eff[i].split(",")[4].split(":")[1]))
    val_f1_eff.append(float(rlines_eff[i].split(",")[5].split(":")[1]))
    train_loss_res.append(float(rlines_res[i].split(",")[2].split(":")[1]))
    val_loss_res.append(float(rlines_res[i].split(",")[3].split(":")[1]))
    val_accu_res.append(float(rlines_res[i].split(",")[4].split(":")[1]))
    val_f1_res.append(float(rlines_res[i].split(",")[5].split(":")[1]))

    
   

# print(loss_log)
# print(loss_svm)
x = np.arange(1,41,1)
plt.rcParams['font.size'] = '25'
# plt.rcParams['font.family'] = 'Palatino'

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=500)

ax.plot(x, train_loss_eff, label="Eff_train_loss", lw=3)
ax.plot(x, val_loss_eff, label="Eff_val_loss", lw=3)
ax.plot(x, train_loss_res, label="Res_train_loss", lw=3)
ax.plot(x, val_loss_res, label="Res_val_loss", lw=3)

ax.set_xlabel("Number of Epoch", size=28)
ax.set_ylabel("Average Loss", size=28)
ax.set_xlim(0, 40)
# ax.set_ylim(0, )
plt.legend(fontsize=20)
plt.tight_layout()

plt.savefig("./images/loss_v3.png")

ax.clear()

ax.plot(x, val_accu_eff, label="EfficientNet_b3", lw=3)
ax.plot(x, val_accu_res, label="ResNet50", lw=3)

ax.set_xlabel("Number of Epoch", size=28)
ax.set_ylabel("Validation Accuracy", size=28)
ax.set_xlim(0, 40)
# ax.set_ylim(80, 97)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("./images/accu_v3.png")


ax.clear()

plt.plot(x, val_f1_eff, label="EfficientNet_b3", lw=3)
plt.plot(x, val_f1_res, label="ResNet50", lw=3)

ax.set_xlabel("Number of Epoch", size=28)
ax.set_ylabel("F1-Score", size=28)
ax.set_xlim(0, 40)
ax.set_ylim(0, 1)
plt.legend(fontsize=20)
plt.tight_layout()

plt.savefig("./images/f1score_v3.png")
