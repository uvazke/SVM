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

def fit(a,x,N,D,etha,lam,lam_update_rate,iter_num,test_data):
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
    # define constant
    N = 100
    N_test = 500
    D = 3
    a = np.zeros(N)

    b = 1
    iter_num = 1000
    etha = 0.0001
    #lam = 1
    lam_update_rate = 0.0001
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

    #make datas
    train_data = np.random.randn(N,D)
    test_data = np.random.randn(N_test,D)
    #[index_t,index_f,t] = true_data(test_data,N_test)
    ACC = 0
    ACC1 = 0
    ACC2 = 0
    ACC3 = 0
    ACC4 = 0
    ACC5 = 0
    ET1 = 0
    ET2 = 0
    ET3 = 0
    ET4 = 0
    ET5 = 0
    learn_num = 100
    for i in range (learn_num):
        #make datas
        train_data = np.random.randn(N,D)
        test_data = np.random.randn(N_test,D)
        [w1,b1,a1,accuracy1, elapsed_time1] = fit(np.zeros(N),train_data,N,D,etha,0,lam_update_rate,iter_num,test_data)
        [w2,b2,a2,accuracy2, elapsed_time2] = fit(np.zeros(N),train_data,N,D,etha,0.5,lam_update_rate,iter_num,test_data)
        [w3,b3,a3,accuracy3, elapsed_time3] = fit(np.zeros(N),train_data,N,D,etha,1,lam_update_rate,iter_num,test_data)
        [w4,b4,a4,accuracy4, elapsed_time4] = fit(np.zeros(N),train_data,N,D,etha,1.5,lam_update_rate,iter_num,test_data)
        [w5,b5,a5,accuracy5, elapsed_time5] = fit(np.zeros(N),train_data,N,D,etha,2.0,lam_update_rate,iter_num,test_data)
        ACC1 = np.append(ACC1, accuracy1)
        ACC2 = np.append(ACC2, accuracy2)
        ACC3 = np.append(ACC3, accuracy3)
        ACC4 = np.append(ACC4, accuracy4)
        ACC5 = np.append(ACC5, accuracy5)
        ET1 = np.append(ET1, elapsed_time1)
        ET2 = np.append(ET2, elapsed_time2)
        ET3 = np.append(ET3, elapsed_time3)
        ET4 = np.append(ET4, elapsed_time4)
        ET5 = np.append(ET5, elapsed_time5)



        print(i)

    ACC1 = ACC1[1:]
    ACC2 = ACC2[1:]
    ACC3 = ACC3[1:]
    ACC4 = ACC4[1:]
    ACC5 = ACC5[1:]

    ET1 = ET1[1:]
    ET2 = ET2[1:]
    ET3 = ET3[1:]
    ET4 = ET4[1:]
    ET5 = ET5[1:]
    ETmean1 = ET1.mean()
    ETSD1 = ET1.std()
    mean1 = ACC1.mean()
    SD1 = ACC1.std()
    ETmean2 = ET2.mean()
    ETSD2 = ET2.std()
    mean2 = ACC2.mean()
    SD2 = ACC2.std()
    ETmean3 = ET3.mean()
    ETSD3 = ET3.std()
    mean3 = ACC3.mean()
    SD3 = ACC3.std()
    ETmean4 = ET4.mean()
    ETSD4 = ET4.std()
    mean4 = ACC4.mean()
    SD4 = ACC4.std()
    ETmean5 = ET5.mean()
    ETSD5 = ET5.std()
    mean5 = ACC5.mean()
    SD5 = ACC5.std()
    ETSD = np.array([ETSD1,ETSD2,ETSD3,ETSD4,ETSD5])
    ETmean = np.array([ETmean1,ETmean2,ETmean3,ETmean4,ETmean5])
    SD = np.array([SD1,SD2,SD3,SD4,SD5])
    SE = SD/10
    mean = np.array([mean1,mean2,mean3,mean4,mean5])
    #save accuracy
    ACC = np.array([ACC1,ACC2,ACC3,ACC4,ACC5]).T
    np.savetxt(ACCdir,ACC,delimiter= ' ')
    size = 1

    plt.figure(figsize=(8 * size, 6 * size))
    x_axis = [i for i in range (iter_num + 1)]

    #plt.plot(x_axis, ACC)
    width = 0.5
    fontsize = 20

    plt.bar([0,1,2,3,4], mean,width, yerr = SE, color='r', edgecolor='k')
    plt.ylim(0.95,1)
    plt.xticks([0,1,2,3,4], ["lam:0", "lam:0.5", "lam:1.0", "lam:1.5", "lam:2.0"], fontsize = fontsize)
    plt.yticks([0.95,0.96,0.97,0.98,0.99,1],fontsize = fontsize)
    plt.ylabel("accuracy",fontsize = fontsize)
    plt.tight_layout()
    print(hms)
    print("SD: " + str(SD))
    print("mean: " +str(mean))
    for i in range (5):
        print("ET")
        print(str(ETmean[i]) +"±"+ str (ETSD[i]))
    for i in range (5):
        print("λ:" + str(0 + i * 0.5) +" "+ str(mean[i]) +"±"+ str (SE[i]))

    seq = np.arange(-3, 3, 0.02)
    """
    plt.figure()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.plot(seq, -(w[0,0] * seq + b) / w[0,1])
    plt.plot(x[index_t,0], x[index_t,1], 'ro')
    plt.plot(x[index_f,0], x[index_f,1], 'bo')
    """
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(seq, seq)
    ax.plot_wireframe(X,Y,-(w[0,0] * X + w[0,1] * Y + b)/ w[0,2])
    #ax.scatter3D(np.ravel(X),np.ravel(Y),-(w[0,0] * X + w[0,1] * Y + b)/ w[0,2])
    ax.plot(test_data[index_t,0],test_data[index_t,1],test_data[index_t,2], 'ro')
    ax.plot(test_data[index_f,0],test_data[index_f,1],test_data[index_f,2], 'bo')
    print(test_data)
    """
    plt.savefig(dirname)
    plt.show()
