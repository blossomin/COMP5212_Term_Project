import matplotlib.pyplot as plt
import sys
import os
import numpy as np


### result 1
file_eff = "./eff_models_v1/log.txt"
file_res = "./res_models_v1/log.txt"


rfile_eff = open(file_eff, "r")
rfile_res = open(file_res, "r")


rlines_eff = rfile_eff.readlines()
rlines_res = rfile_res.readlines()

train_loss_eff, val_loss_eff, val_accu_eff, val_f1_eff = [],[],[],[]
train_loss_res, val_loss_res, val_accu_res, val_f1_res = [], [] ,[], []


print(f"len:{len(rlines_eff)}")
for i in range(len(rlines_eff)):
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

plt.figure()

plt.plot(x, train_loss_eff, label="Eff train loss")
plt.plot(x, val_loss_eff, label="Eff val loss")
plt.plot(x, train_loss_res, label="Res train loss")
plt.plot(x, val_loss_res, label="Res val loss")

# plt.xticks(x)
plt.xlabel("# epoch")
plt.ylabel("average loss")
# plt.title("lr=0.001")

plt.legend()

plt.savefig("./images/loss.png")

plt.figure()

plt.plot(x, val_accu_eff, label="Eff")
plt.plot(x, val_accu_res, label="Res")

plt.xlabel("# epoch")
plt.ylabel("accuracy on validation dataset")

plt.legend()

plt.savefig("./images/accu.png")


plt.figure()

plt.plot(x, val_f1_eff, label="Eff")
plt.plot(x, val_f1_res, label="Res")

plt.xlabel("# epoch")
plt.ylabel("f1 score")

plt.legend()

plt.savefig("./images/f1score.png")
