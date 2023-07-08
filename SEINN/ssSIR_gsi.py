import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize
import matplotlib.dates as dates
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score
# from ipywidgets import interact, widgets
from scipy.integrate import solve_ivp
plt.style.use('seaborn-poster')
matplotlib.rcParams['figure.figsize'] = (10., 6.)
import copy
# import sympy
# %matplotlib inline
import scipy as sp
from scipy.integrate import odeint
import datetime as dt
import timeit
import time
import os
import sys
import time
sys.path.insert(0, '../../Utilities/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scipy.interpolate import CubicSpline
np.set_printoptions(threshold=np.inf)
from prettytable import PrettyTable  
import matplotlib as mpl
import matplotlib.dates as mdates

c_dir =os.getcwd()
path = '/Output_GSI/'
out = c_dir +path
if not os.path.exists(out):
    os.makedirs(out)

np.random.seed(12345)
tf.set_random_seed(12345)
#tf.compat.v1.disable_eager_execution()
class SIRD:
    def __init__(self, I, R, S,T,N0, layers, eps, h, eta, v,alpha, k,sigma, M, dw):
        self.t = T
        self.I = I
        self.R = R
        self.S = S
        self.dw =dw
        self.sigma =sigma 
        self.h =h
        self.eta =eta
        self.v =v
        self.layers = layers
        self.N = N0
        self.M =M
        self.lb = T.min()
        self.ub = T.max()
        #trainable parameters
        self.alpha1 = tf.Variable([alpha], dtype=tf.float32, trainable=False)
        self.kk = tf.Variable([k], dtype=tf.float32, trainable=False)
        #fixed parameters
        self.delta = tf.Variable(0.01,dtype=tf.float32,trainable=False)
        self.b = tf.Variable(1.0,dtype=tf.float32,trainable=False)
        self.d = tf.Variable(0.1,dtype=tf.float32,trainable=False)
        self.gamma = tf.Variable(0.01,dtype=tf.float32,trainable=False)
        self.mu = tf.Variable(0.05,dtype=tf.float32,trainable=False)
        
        
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
        self.dw_tf = tf.placeholder(tf.float32, shape=[None, self.dw.shape[1]])
        # self.D_tf = tf.placeholder(tf.float32, shape=[None, self.D.shape[1]])
        
        #lossess
        self.total_loss =[]
        self.loss_data =[]
        self.loss_phys =[]
        
        self.S_pred, self.I_pred, self.R_pred,_=self.net_sird(self.t_tf)
        
        self.e1, self.e2, self.e3, self.e4= self.net_stochastic_mcmc(self.t_tf)
        
        self.lossPhy = tf.reduce_mean(tf.abs(self.e1)) + tf.reduce_mean(tf.abs(self.e2)) +\
                        tf.reduce_mean(tf.abs(self.e3)) + tf.reduce_mean(tf.abs(self.e4)) 
        self.S_loss = 0
        self.I_loss = 0
        self.R_loss = 0
        iter = 0
        for i in range(len(T)):
            if T[i]%1 == 0:
                self.S_loss += tf.abs(self.S_tf[iter] - self.S_pred[i])
                self.I_loss += tf.abs(self.I_tf[iter] - self.I_pred[i])
                self.R_loss += tf.abs(self.R_tf[iter] - self.R_pred[i])
                iter += 1
        self.lossData = (self.I_loss + self.R_loss + self.S_loss)/iter
        self.loss = (1.0-eps)*self.lossData + eps*self.lossPhy
        
        
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        # Optimizers                                                                                          
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
    
    def net_sird(self, t):
        NN=self.N
        S = self.neural_net1(t, self.weights1, self.biases1)
        I = self.neural_net1(t, self.weights2, self.biases2)
        R = self.neural_net1(t, self.weights3, self.biases3)
        Dw = self.neural_net1(t, self.weights4, self.biases4)
        return S, I, R, Dw
    
    
    def net_stochastic_mcmc(self, t):
        S, I, R, dw=self.net_sird(t)
        b =self.b
        d =self.d
        delta =self.delta
        mu =self.mu
        k =self.kk
        alpha =self.alpha1
        gamma=self.gamma
        h=self.h
        eta =self.eta
        v =self.v
        #define gsi
        gsi =k*S**h*I/(S**h +alpha*I**h)
        M = self.M
        dt = tf.constant(1.0, dtype=tf.float32)
        sigma = self.sigma
        s_list = []
        i_list = []
        r_list = []
        # MCMC simulation
        for i in range(M):
            S_k = S + dt*(b-d*S-gsi +gamma*R -v*eta*S) + sigma*S*tf.math.sqrt(dt) * dw
            I_k = I + dt*(gsi -(d+mu+delta)*I) + sigma*I*tf.math.sqrt(dt) * dw
            R_k = R + dt*(mu*I -(d+gamma)*R+v*eta*S) + sigma*R*tf.math.sqrt(dt) * dw
            s_list.append(S_k)
            i_list.append(I_k)
            r_list.append(R_k)
        e1 = tf.reduce_mean(tf.stack(s_list), axis=0)
        e2 = tf.reduce_mean(tf.stack(i_list), axis=0)
        e3 = tf.reduce_mean(tf.stack(r_list), axis=0)
        e4 = 1.0 - (S+I+R)
        return e1, e2, e3, e4
    

    def callback(self, loss):
        print('Loss: %.3e' % (loss))
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.I_tf: self.I,self.R_tf: self.R,self.S_tf: self.S, self.dw_tf:self.dw}
                   
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_t = self.sess.run(self.loss, tf_dict)
            loss_d = self.sess.run(self.lossData, tf_dict)
            loss_p = self.sess.run(self.lossPhy, tf_dict)
            self.total_loss.append(loss_t)
            self.loss_data.append(loss_d)
            self.loss_phys.append(loss_p)
            
            # Print
            if it % 500 == 0:
                elapsed = time.time() - start_time
                loss_t = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_t,elapsed))
                start_time = time.time()
                 
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        S = self.sess.run(self.S_pred, tf_dict)
        I = self.sess.run(self.I_pred, tf_dict)
        R = self.sess.run(self.R_pred, tf_dict)
        return S,I,R
    
##prepare data

data =pd.read_csv("tndata.csv")
def data_preprocess(data, lb, ub, N0):
    tdat=data.reindex(index=data.index[::-1])
    ic = tdat["TOTAL_CASES"]
    dc = tdat["TOTAL_DEATHS"]
    re = tdat["TOTAL_INACTIVE_RECOVERED"]
    y1, y2 =np.array(ic.values).reshape((-1,1)), np.array(dc.values).reshape((-1,1))
    y3   =np.array(re.values).reshape((-1,1))
    y4 =y1-y3-y2 #infected cases
    I, R, D =y1[lb:ub,:],y3[lb:ub,:], y2[lb:ub,:]
    S =N0-I-R-D
    length =int(ub-lb)
    T = np.arange(0,length).reshape(length,1)
    return S, I, R, D, T

N0 = 6.82*1.e6
lb =281 #change this
ub =431
length=int(ub-lb)
eps =1e-1
eta =0.94
h =2
v =0.01
alpha=0.5
k =0.2
sigma =0.60
M=10
SS, II, RR, DD, T=data_preprocess(data, lb, ub, N0)
    
T = np.arange(0,length - 0.05,0.05)
T = T.reshape(len(T),1)
dw =np.random.normal(size=(150, 1))
layers = [1,60,60,60,60,1] #case 3
def noisy_data(data, sigma):
    data= data + sigma*np.std(data)*np.random.randn(data.shape[0], data.shape[1])
    return data
S2=noisy_data(SS,sigma)
I2 =noisy_data(II, sigma)
R2 =noisy_data(RR, sigma)
model = SIRD(I2/N0,R2/N0,S2/N0, T,1,layers, eps, h, eta, v,alpha, k,sigma, M, dw)
model.train(30000)

#predict output
TT = np.arange(0,length).reshape(length,1)
s_p,i_p,r_p= model.predict(TT)




# case ='c1'
S_original=S2
I_original=I2
R_original=R2
ss =s_p*N0
ii =i_p*N0
#compute gsi
gsi_pred =k*ss**h*ii/(ss**h +alpha*ii**h)
gsi_act =k*S_original**h*I_original/(S_original**h +alpha*I_original**h)


# np.savetxt(out +'S_true_{}_{}_{}_{}.txt'.format(eta, h, v,k),S_original.flatten().tolist())
np.savetxt(out +'I_true_{}_{}_{}.txt'.format(eta, sigma, v),I_original.flatten().tolist())
# np.savetxt(out +'R_true_{}_{}_{}_{}.txt'.format(eta, h, v, k),R_original.flatten().tolist())
# np.savetxt(out +'S_pred_{}_{}_{}_{}.txt'.format(eta, h, v, k),s_p.flatten().tolist())
np.savetxt(out +'I_pred_{}_{}_{}.txt'.format(eta, sigma, v),ii.flatten().tolist())
# np.savetxt(out +'R_pred_{}_{}_{}_{}.txt'.format(eta, h, v, k),r_p.flatten().tolist())
np.savetxt(out +'gsi_pred_{}_{}_{}.txt'.format(eta, sigma, v),gsi_pred.flatten().tolist())
np.savetxt(out +'gsi_act_{}_{}_{}.txt'.format(eta, sigma, v),gsi_act.flatten().tolist())
#loss
##Get errors
I_or =I_original.reshape(-1,)
I_p  =i_p.reshape(-1,)
test_actual=I_or
test_pred =I_p*N0
rmse =np.sqrt(mean_squared_error(test_actual, test_pred))
mape =np.linalg.norm((test_pred-test_actual),2)/np.linalg.norm(test_actual, 2)
#Error Metrics
with open(out+'error_{}_{}_{}.txt'.format(eta, sigma, v), 'w') as f:
    print("Error metrics for  eta={} and sigma ={}, ".format(eta, sigma, v), file=f)
    print("=======================================================\n", file=f)
    print('RMSE',rmse, file=f)
    print('MAPE',mape, file=f)
    print("=======================================================\n", file=f)
  
