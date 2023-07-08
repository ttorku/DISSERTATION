import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize
import matplotlib.dates as dates
import copy
import math
import scipy as sp
from scipy.integrate import odeint
import datetime as dt
import timeit
import time
import os
import sys
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
sys.path.insert(0, '../../Utilities/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scipy.interpolate import CubicSpline
np.set_printoptions(threshold=np.inf)
from prettytable import PrettyTable  
import matplotlib as mpl
import matplotlib.dates as mdates


np.random.seed(12345)
tf.set_random_seed(12345)
#tf.compat.v1.disable_eager_execution()
class SIR:
    def __init__(self, I, R, S,T,N0, layers, eta, v,alpha1, alpha2, dW, n_mc):
        self.t = T
        self.I = I
        self.R = R
        self.S = S
        self.dW=dW
        self.n_mc =n_mc
        self.alpha1, self.alpha2 =alpha1, alpha2
        self.eta =eta
        self.N0=N0
        self.v =v
        self.layers = layers
        self.lb = T.min()
        self.ub = T.max()
        #trainable parameters
        self.beta = tf.Variable(0.5,dtype=tf.float32,trainable=True)
        self.gamma = tf.Variable(0.0,dtype=tf.float32,trainable=True)
        
        # self.gamma = tf.Variable([1], constraint=lambda x: tf.abs(x),dtype=tf.float32)
        # Initialize NN
        self.weights1, self.biases1 = self.initialize_NN(self.layers)
        self.weights2, self.biases2 = self.initialize_NN(self.layers)
        self.weights3, self.biases3 = self.initialize_NN(self.layers)
        self.weights4, self.biases4 = self.initialize_NN(self.layers)
     
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
        self.R_tf = tf.placeholder(tf.float32, shape=[None, self.R.shape[1]])
        self.S_tf = tf.placeholder(tf.float32, shape=[None, self.S.shape[1]])
        self.dW_tf = tf.placeholder(tf.float32, shape=[None, self.dW.shape[1]])
        
        #lossess
        self.total_loss =[]
        self.loss_data =[]
        self.loss_res =[]
        
        self.S_pred, self.I_pred, self.R_pred,_=self.net_sir(self.t_tf)
        
        self.l_m, self.s_list, self.i_list, self.r_list= self.euler_maruyama_loss(self.t_tf, self.n_mc) #self.e4
        
        self.lossRes = tf.reduce_mean(tf.square(self.l_m)) 
        self.S_loss =tf.reduce_mean(tf.square(self.S_tf - self.S_pred))
        self.I_loss =tf.reduce_mean(tf.square(self.I_tf - self.I_pred))
        self.R_loss =tf.reduce_mean(tf.square(self.R_tf - self.R_pred))
        
        self.lossData =self.I_loss + self.R_loss + self.S_loss
        # self.S_loss = 0
        # self.I_loss = 0
        # self.R_loss = 0
        # iter = 0
        # for i in range(len(T)):
        #     if T[i]%1 == 0:
        #         self.S_loss += tf.square(self.S_tf[iter] - self.S_pred[i])
        #         self.I_loss += tf.square(self.I_tf[iter] - self.I_pred[i])
        #         self.R_loss += tf.square(self.R_tf[iter] - self.R_pred[i])
        #         iter += 1
        # self.lossData = (self.I_loss + self.R_loss + self.S_loss)/iter
        alpha1, alpha2 =self.alpha1, self.alpha2
        self.loss = alpha1*self.lossData + alpha2*self.lossRes
        '''
        self.loss = tf.reduce_mean(tf.abs(self.I_tf - self.I_pred)) + tf.reduce_mean(tf.abs(self.R_pred - self.R_tf)) +\
                    tf.reduce_mean(tf.abs(self.S_tf - self.S_pred)) +\
                    tf.reduce_mean(tf.abs(self.E1_pred))+ tf.reduce_mean(tf.abs(self.E2_pred)) +\
                    tf.reduce_mean(tf.abs(self.E3_pred)) + tf.reduce_mean(tf.abs(self.E4_pred)) 
        '''
        
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(1e-4)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        # Optimizers
        '''
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})    
        '''                                                                                          
        init = tf.global_variables_initializer()
        self.sess.run(init)
         
        
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers)
        
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        return Y

    def neural_net1(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
        return Y
    
    def net_sir(self, t):
        SIRd = self.neural_net1(t, self.weights1, self.biases1)
        S, I, R, Dw =SIRd[:,0:1], SIRd[:,1:2], SIRd[:,2:3], SIRd[:,3:4]
        # I = self.neural_net1(t, self.weights2, self.biases2)
        # R = self.neural_net1(t, self.weights3, self.biases3)
        # Dw = self.neural_net1(t, self.weights4, self.biases4)
        return S, I, R, Dw
    
     # Define the SIR model equations
    def SIR_equations(self, S, I, R, beta, gamma, N, dW,sigma, dt,eta, v):
        dS = -beta * S * I / N+v*eta*S/N + sigma*S*math.sqrt(dt) * dW
        dI = beta * S * I / N - gamma * I + sigma*I*math.sqrt(dt) * dW
        dR = gamma * I-v*eta*S/N + sigma*R*math.sqrt(dt) * dW
        return dS, dI, dR
    
    # Define the Euler-Maruyama discretization loss
    def euler_maruyama_loss(self, t, n_mc):
        N=self.N0  
        v, eta=self.v, self.eta
        beta, gamma =self.beta, self.gamma
        dt =1.0
        S_list=[]
        I_list=[]
        R_list=[]
        loss =[]
        for _ in range(n_mc):
            S, I, R, dW=self.net_sir(t)
            dS, dI, dR = self.SIR_equations(S, I, R, beta, gamma, N, dW,sigma, dt,eta, v)
            S_new = S + dt * dS
            I_new = I + dt * dI
            R_new = R + dt * dR
            e1 =S_new - S
            e2 =I_new - I
            e3 =R_new - R
            S_list.append(S)
            I_list.append(I)
            R_list.append(R)
            l =tf.reduce_mean(tf.square(e1))+tf.reduce_mean(tf.square(e2))+tf.reduce_mean(tf.square(e3))
            loss.append(l)
        l_m =tf.stack(tf.reduce_mean(loss), axis=0)
        return l_m, S_list, I_list, R_list
    

    def callback(self, loss):
        print('Loss: %.3e' % (loss))
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.I_tf: self.I,self.R_tf: self.R,self.S_tf: self.S, self.dW_tf:self.dW}
                   
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_t = self.sess.run(self.loss, tf_dict)
            loss_d = self.sess.run(self.lossData, tf_dict)
            loss_p = self.sess.run(self.lossRes, tf_dict)
            self.total_loss.append(loss_t)
            self.loss_data.append(loss_d)
            self.loss_res.append(loss_p)
            
            # Print
            if it % 500 == 0:
                elapsed = time.time() - start_time
                loss_t = self.sess.run(self.loss, tf_dict)
                # loss_d = self.sess.run(self.lossData, tf_dict)
                # loss_p = self.sess.run(self.lossPhy, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_t,elapsed))
                start_time = time.time()
        '''
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
        '''              
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        S_l = self.sess.run(self.s_list, tf_dict)
        I_l = self.sess.run(self.i_list, tf_dict)
        R_l = self.sess.run(self.r_list, tf_dict)
        return S_l,I_l, R_l
    
c_dir =os.getcwd()
path = '/Output/'   #for sir models
out = c_dir +path
if not os.path.exists(out):
    os.makedirs(out)
    
data =pd.read_csv("tndata.csv")
def data_preprocess(data, lb, ub, N0, npoints):
    tdat=data.reindex(index=data.index[::-1])
    ic = tdat["TOTAL_CONFIRMED"]
    re = tdat["TOTAL_INACTIVE_RECOVERED"]
    # y1, y2 =np.array(ic.values).reshape((-1,1)), np.array(re.values).reshape((-1,1))
    y1, y2 =np.array(ic.values), np.array(re.values)
    I, R=y1[lb:ub],y2[lb:ub]
    S =N0-I-R
    length =int(ub-lb)
    T = np.arange(0,length)
    s1 =CubicSpline(T,S)
    s2 =CubicSpline(T,I)
    s3 =CubicSpline(T,R)
    tt=np.linspace(0,length, npoints)
    x, y1, y2, y3 =tt, s1(tt), s2(tt),s3(tt)
    t, S1, I1, R1 =x.reshape((-1,1)), y1.reshape((-1,1)), y2.reshape((-1,1)),y3.reshape((-1,1))
    return S1, I1, R1, t

def noisy_data(data, sigma):
    data= data + sigma*np.std(data)*np.random.randn(data.shape[0], data.shape[1])
    return data

N0 = 6.82*1.e6
lb =281 #change this
ub =431
length=int(ub-lb)
v, eta, sigma =0.1, 0.94, 0.60
npoints=3000
SS, II, RR, T=data_preprocess(data, lb, ub, N0, npoints)
S2=noisy_data(SS,sigma)
I2 =noisy_data(II, sigma)
R2 =noisy_data(RR, sigma)
dW=np.random.normal(size=(npoints, 1))    
# T = np.arange(0,length - 0.05,0.05)
# T = T.reshape(len(T),1)

def data_preprocess1(data, lb, ub, N0):
    tdat=data.reindex(index=data.index[::-1])
    ic = tdat["TOTAL_CONFIRMED"]
    re = tdat["TOTAL_INACTIVE_RECOVERED"]
    y1, y2 =np.array(ic.values).reshape((-1,1)), np.array(re.values).reshape((-1,1))
    I, R=y1[lb:ub,:],y2[lb:ub,:]
    S =N0-I-R
    length =int(ub-lb)
    T = np.arange(0,length).reshape(length,1)
    return S, I, R, T
_, _, _, t1=data_preprocess1(data, lb, ub, N0)
alpha1 =1.0
alpha2 =1.0
epochs =40000

n_mc=10
layers = [1,64,64,64,64,4]
model = SIR(I2/N0, R2/N0, S2/N0,T,1.0, layers, eta, v, alpha1, alpha2, dW, n_mc)
model.train(epochs)  #for Adam
S_l, I_l, R_l = model.predict(T)
v1 =np.array(S_l)
v2 =np.array(I_l)
v3 =np.array(R_l)
S_pred_m =np.mean(v1, axis=0)
I_pred_m =np.mean(v2, axis=0)
R_pred_m =np.mean(v3, axis=0)
S_pred_s =np.std(v1, axis=0)
I_pred_s =np.std(v2, axis=0)
R_pred_s =np.std(v3, axis=0)
beta_v1=model.sess.run(model.beta)
gamma_v1=model.sess.run(model.gamma)
beta, gamma =round(beta_v1, 5), round(gamma_v1,5)
    
np.savetxt(out +'S_stoM_{}_{}_{}.txt'.format(sigma, eta, v), S_pred_m.flatten().tolist())
np.savetxt(out +'I_stoM_{}_{}_{}.txt'.format(sigma, eta, v), I_pred_m.flatten().tolist())
np.savetxt(out +'R_stoM_{}_{}_{}.txt'.format(sigma,eta, v), R_pred_m.flatten().tolist())

np.savetxt(out +'S_stoS_{}_{}_{}.txt'.format(sigma, eta, v), S_pred_s.flatten().tolist())
np.savetxt(out +'I_stoS_{}_{}_{}.txt'.format(sigma, eta, v),I_pred_s.flatten().tolist())
np.savetxt(out +'R_stoS_{}_{}_{}.txt'.format(sigma,eta, v),R_pred_s.flatten().tolist())
##Calculate the error metrics for stochastic for Infected
I_p =I_pred_m.reshape(-1,)
I_or =I2.reshape(-1,)
test_actual=I_or
test_pred =I_p*N0
rmse =np.sqrt(mean_squared_error(test_actual, test_pred))
mape =np.linalg.norm((test_pred-test_actual),2)/np.linalg.norm(test_actual, 2)   
with open(out+'error1_{}_{}_{}.txt'.format(sigma,eta, v), 'w') as f:
    print("Error metrics for  sigma {}, eta {} and v {}".format(sigma,eta, v), file=f)
    print("=======================================================\n", file=f)
    print('RMSE',rmse, file=f)
    print('MAPE',mape, file=f)
    print('beta',beta, file=f)
    print('gamma',gamma, file=f)
    print("=======================================================\n", file=f)