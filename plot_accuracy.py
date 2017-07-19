import time
import numpy as np
import matplotlib.pyplot as plt

import scipy.fftpack as scp

def mm2inch(x):
    return x/25.4



ymd = "2017719" #埋めて
hms1 = "205118" #埋めて
hms2 = "205149" #埋めて
hms3 = "205225" #埋めて
hms4 = "205310" #埋めて
hms5 = "195514" #埋めて
dirname = "../datas/SVM/%s" % ymd
figname = "/accuracygraph%s.png" % hms1
ACCdir1 = dirname + "/accuracy%s.csv" % hms1
ACCdir2 = dirname + "/accuracy%s.csv" % hms2
ACCdir3 = dirname + "/accuracy%s.csv" % hms3
ACCdir4 = dirname + "/accuracy%s.csv" % hms4
ACCdir5 = dirname + "/accuracy%s.csv" % hms5
figdir = dirname + figname

data1 = np.loadtxt(ACCdir1)
data2 = np.loadtxt(ACCdir2)
data3 = np.loadtxt(ACCdir3)
data4 = np.loadtxt(ACCdir4)
data5 = np.loadtxt(ACCdir5)

size = 1
iter_num = 10000
plt.figure(figsize=(8 * size, 6 * size))
x_axis = [i for i in range (iter_num + 1)]
plt.plot(x_axis, data1,label = "lam = 0")
plt.plot(x_axis, data2,label = "lam = 1")
plt.plot(x_axis, data3,label = "lam = 2")
plt.plot(x_axis, data4,label = "lam = 3")
plt.plot(x_axis, data5,label = "lam = 10")
plt.ylim(0.5,1)
plt.xlabel("learning number (times)")
plt.ylabel("accuracy")
plt.legend(loc= "lower right")

plt.savefig(figdir)
plt.show()
