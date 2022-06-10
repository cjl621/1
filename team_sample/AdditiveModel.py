import csv
import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def vshift(arr, s):
    for i in range(len(arr)):
        arr[i] -= s
        if arr[i] < 0:
            arr[i] += 24
    return arr

class LagFitter:
    def __init__(self, N):
        self.N = N
        self.t = [0]*23*N
        self.lag = [0]*23*N
        self.cnt = 0

    def genLagData(self, row):
        if self.cnt >= self.N*23:
            return
        
        lag = [0]*23
        raw = [0]*23

        raw[0] = int(row[0])
        for h in range(1,23):
            raw[h] = int(row[h])
            self.t[self.cnt+h] = h
            self.lag[self.cnt+h] = raw[h]-raw[h-1]
        
        self.cnt += 23

    def fit(self, t0, t1):
        # Specifying 3 knots
        transformed_t = dmatrix("bs(self.t, knots=(t0,t1), degree=3, include_intercept=False)",
                        {"self.t": self.t}, return_type='dataframe')

        # Build a regular linear model from the splines
        fit1 = sm.GLM(self.lag, transformed_t).fit()
        print(fit1.params)

class Viewer:
    def __init__(self, N):
        self.epslon = 0
        self.t = ['8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','1','2','3','4','5','6','7']
        self.N = N # sample number
        self.ave_cnt = 0
        self.ave = [0]*N

        self.cnt = 0

    def plotRow(self, row):
        raw = [0]*24
        for h in range(0,23):
            raw[h] = int(row[h])
        plt.plot(self.t, raw, ls='-')

    def plotAveZeroRow(self, row):
        raw = [0]*24
        total = 0
        for h in range(0,23):
            raw[h] = int(row[h])
            total += raw[h]

        for h in range(0,23):
            raw[h] -= total/24.0

        plt.plot(self.t, raw , ls='-')

    def plotRowAve(self, row):
        if self.ave_cnt >= self.N:
            return

        raw = 0
        for h in range(0,23):
            raw += float(row[h])      

        self.ave[self.ave_cnt] = raw/24.0 
        self.ave_cnt += 1

    def plotlag(self, row):
        if self.cnt >= self.N:
            return

        lag = [0]*23
        raw = [0]*23

        raw[0] = int(row[0])
        for h in range(1,22):
            raw[h] = int(row[h])
            lag[h-1] = (raw[h]-raw[h-1])/2
        lag[22] = raw[22]/2

        t=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
        t1=[0]*23
        for h in range(1,22):
            t1[h] = 0.5

        plt.quiver(t, raw, t1, lag, linewidth=None, color='g')

        self.cnt += 1

    def show(self):
        if self.ave_cnt > 0:
            t = [0]*self.ave_cnt
            for i in range(0,self.ave_cnt):
                t[i] = i

            sign_neg = 0
            for i in range(1,self.ave_cnt):
                if self.ave[i] - self.ave[i-1] < 0 :
                    sign_neg += 1
            print(sign_neg)

            t = np.array(t).reshape((-1, 1))
            model = LinearRegression().fit(t, self.ave)
            print(f"seasonal intercept: {model.intercept_}")
            print(f"seasonal slope: {model.coef_}")

            t = [0]*self.ave_cnt
            y = [0]*self.ave_cnt
            for i in range(0,self.ave_cnt):
                t[i] = i
                y[i] = i*model.coef_+model.intercept_

            plt.plot(t, y, ls='-.')
            plt.scatter(t, self.ave)

        plt.show()

class Classifier:
    def __init__(self, b0_arr, b1_arr):
        self.b0_arr = b0_arr
        self.b1_arr = b1_arr

        self.err0 = [0]*len(b0_arr)*len(b1_arr)
        self.n0 = [1]*len(b0_arr)*len(b1_arr)
        self.err1 = [0]*len(b0_arr)*len(b1_arr)
        self.n1 = [1]*len(b0_arr)*len(b1_arr)
        self.err2 = [0]*len(b0_arr)*len(b1_arr)
        self.n2 = [1]*len(b0_arr)*len(b1_arr)

        self.err_best = 1
        self.b0_best = 0
        self.b1_best = 0
        self.classes_best = (0,0,0,0,0,0)

        self.t_max = [0]*30
        self.t_min = [0]*30
        self.N = 0

    def processSample(self, row):     
        # comupute lag1
        lag1 = [0]*24
        raw = 0
        last_raw = int(row[0])
        for h in range(1,23):
            raw = int(row[h])
            lag1[h-1] = raw - last_raw
            last_raw = raw

        # to show an unrobust algorithm        
        max = -100
        min = 100
        max_n = 0
        min_n = 0
        for h in range(0,23):
            raw = int(row[h])
            if raw > max:
                max = raw
                max_n = h
            elif raw < min:
                min = raw
                min_n = h
        self.t_max[self.N] = max_n
        self.t_min[self.N] = min_n
        self.N += 1

        # classify lag1 by all possible boundaries
        for i in range(len(self.b0_arr)):
            for j in range(len(self.b1_arr)):
                t0 = self.b0_arr[i]
                t1 = self.b1_arr[j]
                buf_pos = i*len(self.b1_arr)+j
                for h in range(0,23):
                    if h <= t0:
                        self.n0[buf_pos] += 1
                        if lag1[h] < 0:
                            self.err0[buf_pos] += 1
                    elif h <= t1:
                        self.n1[buf_pos] += 1
                        if lag1[h] > 0:
                            self.err1[buf_pos] += 1
                    else:
                        self.n2[buf_pos] += 1
                        if lag1[h] < 0:
                            self.err2[buf_pos] += 1                        
    def missErrSummary(self):
        for i in range(len(self.b0_arr)):
            for j in range(len(self.b1_arr)):
                buf_pos = i*len(self.b1_arr)+j
                err = (self.err0[buf_pos] + self.err2[buf_pos])/(self.n0[buf_pos]+self.n2[buf_pos]) + self.err1[buf_pos]/self.n1[buf_pos]
                if self.err_best > err:
                    self.err_best = err
                    self.b0_best = i
                    self.b1_best = j

        best_pos = (self.b0_best*len(self.b0_arr) + self.b1_best)
        self.b0_best = self.b0_arr[self.b0_best]
        self.b1_best = self.b1_arr[self.b1_best]
        self.classes_best = (self.err0[best_pos]+self.err2[best_pos], self.n0[best_pos]+self.n2[best_pos], self.err1[best_pos], self.n1[best_pos])

    def giniSummary(self):
        for i in range(len(self.b0_arr)):
            for j in range(len(self.b1_arr)):
                buf_pos = i*len(self.b1_arr)+j
                p0 = (self.err0[buf_pos] + self.err2[buf_pos])/(self.n0[buf_pos]+self.n2[buf_pos])
                p1 = self.err1[buf_pos]/self.n1[buf_pos]
                err = p0*p1
                if self.err_best > err:
                    self.err_best = err
                    self.b0_best = i
                    self.b1_best = j

        best_pos = (self.b0_best*len(self.b0_arr) + self.b1_best)
        self.b0_best = self.b0_arr[self.b0_best]
        self.b1_best = self.b1_arr[self.b1_best]
        self.classes_best = (self.err0[best_pos]+self.err2[best_pos], self.n0[best_pos]+self.n2[best_pos], self.err1[best_pos], self.n1[best_pos])
    
    def maxminSummary(self):
        plt.plot(self.t_max)
        plt.plot(self.t_min)
        plt.show()

    def print(self):
        print( self.err_best/2,  self.b0_best+8,  self.b1_best-16,  self.classes_best)

with open('weaher.txt', 'r') as csvfile:
    reader = csv.reader(csvfile)

    # record start at 8 o'clock so shift the value of boundry


    class1 = Classifier(vshift([11,12,13,14,15],8), vshift([3,4,5,6],8))
    lagger = LagFitter(22)
    viewer = Viewer(22)

    # read stream of file
    for row in reader:
        class1.processSample(row)
        lagger.genLagData(row)
        #viewer.plotlag(row)
        viewer.plotRowAve(row)
        #viewer.plotRow(row)
        #viewer.plotAveZeroRow(row)

    class1.missErrSummary()
    class1.print()
    class1.giniSummary();
    class1.print()

    viewer.show()

    lagger.fit(5, 21)

    #class1.maxminSummary()

    

    
