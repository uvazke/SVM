import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

import datetime
import os
import time



def axt (a,x,t):
    at = a*t
    return np.dot(at[:,np.newaxis].T,x)

def calculate_matrix(a, x, t, N, etha, lam, lam_update_rate):
    delta = (1.0 - (np.dot(axt(a,x,t), x.T) * t).T  - (lam/2.0) * (a*t*t)[:,np.newaxis])
    a[:,np.newaxis] += etha * delta
    return [a, lam]

def calculate_respective(a, x, t, N, etha, lam, lam_update_rate):
    for j in range (N):
        a[j] += etha * (1.0 - np.dot(axt(a,x,t), x[j])*t[j] - lam * np.dot(a,t) * t[j])
    lam += lam * lam_update_rate
    return [a, lam]

def true_data(data,N,random = False):
    if random == True:
        r = np.random.randint(0,2,N)
        index_t = r == 1
    else:
        index_t = data[:,0]>=data[:,1]

    index_f = index_t == False
    t = np.array([(-1) for i in range (N)])
    t[index_t] = 1
    return [index_t,index_f,t]

def fit_and_test(a,x,N,D,etha,lam,lam_update_rate,iter_num,test_data):
    [index_t,index_f,t] = true_data(x,N)
    L = a.sum() - np.dot(axt(a,x,t),axt(a,x,t).T)/2
    max_a = a
    N_test = 500
    ACC = 0
    start = time.time()

    for i in range (iter_num):
        [a,lam] = calculate_matrix(a,x,t,N,etha,lam,lam_update_rate)
        L_new = a.sum() - np.dot(axt(a,x,t),axt(a,x,t).T)/2

        if L_new >= L:
            max_L_num = i
            L = L_new
            max_a = a
    elapsed_time = time.time() - start

    w = axt(max_a,x,t)
    b = (t - np.dot(w,x.T)).mean()
    ACC = put_accuracy(N_test,test_data,w,b)
    #print("i:" + str(max_L_num))
    return [w, b, max_a, ACC, elapsed_time]

def put_accuracy(N_test,test_data,w,b):
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    [index_t,index_f,t] = true_data(test_data,N_test)
    for i in range (N_test):
        y = np.dot(test_data[i],w.T) + b
        if y >= 0:
            if t[i] == 1:
                TP += 1.0
            else:
                TN += 1.0
        else:
            if t[i] == 1:
                FN += 1.0
            else:
                FP += 1.0
    accuracy = (TP + FP) / (N_test)
    return accuracy

if __name__ == "__main__":
    # define constant --------------------------------------
    N = 100
    N_test = 500
    D = 3

    b = 1
    iter_num = 1000
    etha = 0.0001
    #lam = 1
    lam_update_rate = 0.0001

    ACC1 = 0
    ACC2 = 0
    ACC3 = 0
    ACC4 = 0
    ACC5 = 0

    learn_num = 100
    #-------------------------------------------------------
    #make directory, file for saving datas -----------------
    d = datetime.datetime.today()

    ymd = str(d.year)+str(d.month)+str(d.day)
    hms = str(d.hour)+str(d.minute)+str(d.second)

    dirname = "../datas/SVM/%s/" %ymd
    figname = "%snumbergraph.png" % hms
    ACCdir = "accuracySE%s.csv" % hms
    ACCdir = dirname + ACCdir

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    dirname = dirname + figname
    #-------------------------------------------------------
    #fit & test -------------------------------------------------
    for i in range (learn_num):
        #make datas
        train_data = np.random.randn(N,D)
        test_data = np.random.randn(N_test,D)
        [w1,b1,a1,accuracy1, elapsed_time1] = fit_and_test(np.zeros(N),train_data,N,D,etha,0,lam_update_rate,iter_num,test_data)
        [w2,b2,a2,accuracy2, elapsed_time2] = fit_and_test(np.zeros(N),train_data,N,D,etha,0.5,lam_update_rate,iter_num,test_data)
        [w3,b3,a3,accuracy3, elapsed_time3] = fit_and_test(np.zeros(N),train_data,N,D,etha,1,lam_update_rate,iter_num,test_data)
        [w4,b4,a4,accuracy4, elapsed_time4] = fit_and_test(np.zeros(N),train_data,N,D,etha,1.5,lam_update_rate,iter_num,test_data)
        [w5,b5,a5,accuracy5, elapsed_time5] = fit_and_test(np.zeros(N),train_data,N,D,etha,2.0,lam_update_rate,iter_num,test_data)
        ACC1 = np.append(ACC1, accuracy1)
        ACC2 = np.append(ACC2, accuracy2)
        ACC3 = np.append(ACC3, accuracy3)
        ACC4 = np.append(ACC4, accuracy4)
        ACC5 = np.append(ACC5, accuracy5)

    ACC1 = ACC1[1:]
    ACC2 = ACC2[1:]
    ACC3 = ACC3[1:]
    ACC4 = ACC4[1:]
    ACC5 = ACC5[1:]

    mean1 = ACC1.mean()
    SD1 = ACC1.std()

    mean2 = ACC2.mean()
    SD2 = ACC2.std()

    mean3 = ACC3.mean()
    SD3 = ACC3.std()

    mean4 = ACC4.mean()
    SD4 = ACC4.std()

    mean5 = ACC5.mean()
    SD5 = ACC5.std()

    SD = np.array([SD1,SD2,SD3,SD4,SD5])
    SE = SD/math.sqrt(learn_num)
    mean = np.array([mean1,mean2,mean3,mean4,mean5])

    #save accuracy
    ACC = np.array([ACC1,ACC2,ACC3,ACC4,ACC5]).T

    np.savetxt(ACCdir,ACC,delimiter= ' ')
    for i in range (5):
        print("λ:" + str(0 + i * 0.5) +" "+ str(mean[i]) +"±"+ str (SE[i]))


    #make graph-------------------------------------------
    width = 0.5
    fontsize = 20
    size = 1
    plt.figure(figsize=(8 * size, 6 * size))
    x_axis = [i for i in range (iter_num + 1)]
    plt.bar([0,1,2,3,4], mean,width, yerr = SE, color='r', edgecolor='k')
    plt.ylim(0.95,1)
    plt.xticks([0,1,2,3,4], ["lam:0", "lam:0.5", "lam:1.0", "lam:1.5", "lam:2.0"], fontsize = fontsize)
    plt.yticks([0.95,0.96,0.97,0.98,0.99,1],fontsize = fontsize)
    plt.ylabel("accuracy",fontsize = fontsize)
    plt.tight_layout()


    plt.savefig(dirname)
    plt.show()
    #------------------------------------------------------
