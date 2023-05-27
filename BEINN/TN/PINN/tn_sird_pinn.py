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
path = '/Figures/'
out = c_dir +path
if not os.path.exists(out):
    os.makedirs(out)

np.random.seed(12345)
tf.set_random_seed(12345)
#tf.compat.v1.disable_eager_execution()
class SIRD:
    def __init__(self, I, R, S,D, T,N0, layers):
        self.t = T
        self.I = I
        self.R = R
        self.S = S
        self.D = D
        self.layers = layers
        self.N = N0
        self.lb = T.min()
        self.ub = T.max()
        # self.gamma = tf.Variable([1], constraint=lambda x: tf.abs(x),dtype=tf.float32)
        # Initialize NN
        self.weights1, self.biases1 = self.initialize_NN(self.layers)
        self.weights2, self.biases2 = self.initialize_NN(self.layers)
        self.weights3, self.biases3 = self.initialize_NN(self.layers)
        self.weights4, self.biases4 = self.initialize_NN(self.layers)
        self.weights5, self.biases5 = self.initialize_NN(self.layers)
        self.weights6, self.biases6 = self.initialize_NN(self.layers)
        self.weights7, self.biases7 = self.initialize_NN(self.layers)
        self.weights8, self.biases8 = self.initialize_NN(self.layers)
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.I_tf = tf.placeholder(tf.float32, shape=[None, self.I.shape[1]])
        self.R_tf = tf.placeholder(tf.float32, shape=[None, self.R.shape[1]])
        self.S_tf = tf.placeholder(tf.float32, shape=[None, self.S.shape[1]])
        self.D_tf = tf.placeholder(tf.float32, shape=[None, self.D.shape[1]])
        
        #lossess
        self.total_loss =[]
        self.loss_data =[]
        self.loss_phys =[]
        self.total_time=[]
        self.S_pred, self.I_pred, self.R_pred,self.D_pred =self.net_sird(self.t_tf)
        
        self.beta_pred,self.gamma_pred, self.delta_pred, self.compliance_pred =self.net_param(self.t_tf)
        
        self.e1, self.e2, self.e3, self.e4, self.e5 = self.net_residual(self.t_tf)
        
        self.lossPhy = tf.reduce_mean(tf.abs(self.e1)) + tf.reduce_mean(tf.abs(self.e2)) +\
                        tf.reduce_mean(tf.abs(self.e3)) + tf.reduce_mean(tf.abs(self.e4)) +\
                        tf.reduce_mean(tf.abs(self.e5))
        self.S_loss = 0
        self.I_loss = 0
        self.R_loss = 0
        self.D_loss = 0
        iter = 0
        for i in range(len(T)):
            if T[i]%1 == 0:
                self.S_loss += tf.abs(self.S_tf[iter] - self.S_pred[i])
                self.I_loss += tf.abs(self.I_tf[iter] - self.I_pred[i])
                self.R_loss += tf.abs(self.R_tf[iter] - self.R_pred[i])
                self.D_loss += tf.abs(self.D_tf[iter] - self.D_pred[i])
                iter += 1
        self.lossData = (self.I_loss + self.R_loss + self.S_loss+self.D_loss)/iter
        self.loss = self.lossData + self.lossPhy
        '''
        self.loss = tf.reduce_mean(tf.abs(self.I_tf - self.I_pred)) + tf.reduce_mean(tf.abs(self.R_pred - self.R_tf)) +\
                    tf.reduce_mean(tf.abs(self.S_tf - self.S_pred)) +\
                    tf.reduce_mean(tf.abs(self.E1_pred))+ tf.reduce_mean(tf.abs(self.E2_pred)) +\
                    tf.reduce_mean(tf.abs(self.E3_pred)) + tf.reduce_mean(tf.abs(self.E4_pred)) 
        '''
        
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
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
    
    def net_sird(self, t):
        NN=self.N
        S = self.neural_net1(t, self.weights1, self.biases1)
        I = self.neural_net1(t, self.weights2, self.biases2)
        R = self.neural_net1(t, self.weights3, self.biases3)
        D = self.neural_net1(t, self.weights4, self.biases4)
        return S, I, R, D
    
    def net_param(self, t):
        NN=self.N
        beta= self.neural_net1(t, self.weights5, self.biases5)
        gamma = self.neural_net1(t,self.weights6,self.biases6)
        delta = self.neural_net1(t,self.weights7,self.biases7)
        compliance = self.neural_net1(t,self.weights8,self.biases8)
        return beta, gamma, delta, compliance
        

    def net_residual(self, t):
        NN=self.N  
        S, I, R, D =self.net_sird(t)
        # print(S.shape)
        beta, gamma, delta, compliance =self.net_param(t) 
        St = tf.gradients(S, t)[0]
        It = tf.gradients(I, t)[0]  
        Rt = tf.gradients(R, t)[0]
        Dt = tf.gradients(D, t)[0]
        
        # print(St.shape)
        
        e1 = St + beta*(1.0-compliance)*S*I
        e2 = It - beta*(1.0-compliance)*S*I + gamma*I +delta*I
        e3 = Rt - gamma*I
        e4 = Dt - delta*I
        e5 = 1.0 - (S+I+R+D)
        return e1, e2, e3, e4, e5

    def callback(self, loss):
        print('Loss: %.3e' % (loss))
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.I_tf: self.I,self.R_tf: self.R,self.S_tf: self.S, self.D_tf: self.D}
                   
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_t = self.sess.run(self.loss, tf_dict)
            loss_d = self.sess.run(self.lossData, tf_dict)
            loss_p = self.sess.run(self.lossPhy, tf_dict)
            self.total_loss.append(loss_t)
            self.loss_data.append(loss_d)
            self.loss_phys.append(loss_p)
            elapsed = time.time() - start_time
            self.total_time.append(elapsed)
            
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
        S = self.sess.run(self.S_pred, tf_dict)
        I = self.sess.run(self.I_pred, tf_dict)
        R = self.sess.run(self.R_pred, tf_dict)
        D = self.sess.run(self.D_pred, tf_dict)
        beta = self.sess.run(self.beta_pred,tf_dict)
        gamma = self.sess.run(self.gamma_pred,tf_dict)
        delta = self.sess.run(self.delta_pred,tf_dict)
        compliance = self.sess.run(self.compliance_pred,tf_dict)
        return S,I,R,D, beta, gamma, delta, compliance
    
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
lb =19 #change this
ub =280
length=int(ub-lb)
SS, II, RR, DD, T=data_preprocess(data, lb, ub, N0)
    
T = np.arange(0,length - 0.05,0.05)#here I used Nt with 0.1 stepsize
T = T.reshape(len(T),1)
layers = [1,60,60,60,60,1] #case 3
start_time = time.time()
model = SIRD(II/N0,RR/N0,SS/N0,DD/N0, T,1,layers)
model.train(40000)
end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time
# Convert elapsed time to a readable format
formatted_time = str(dt.timedelta(seconds=elapsed_time))

#predict output
TT = np.arange(0,length).reshape(length,1)
s_p,i_p,r_p,d_p,beta_p, gamma_p, delta_p, compliance_p= model.predict(TT)

case ='c1'
S_original=SS/N0
I_original=II/N0
R_original=RR/N0
D_original=DD/N0

np.savetxt(out +'S_truepinn_{}.txt'.format(case),S_original.flatten().tolist())
np.savetxt(out +'I_truepinn_{}.txt'.format(case),I_original.flatten().tolist())
np.savetxt(out +'R_truepinn_{}.txt'.format(case),R_original.flatten().tolist())
np.savetxt(out +'D_truepinn_{}.txt'.format(case),D_original.flatten().tolist())
np.savetxt(out +'S_predpinn_{}.txt'.format(case),s_p.flatten().tolist())
np.savetxt(out +'I_predpinn_{}.txt'.format(case),i_p.flatten().tolist())
np.savetxt(out +'R_predpinn_{}.txt'.format(case),r_p.flatten().tolist())
np.savetxt(out +'D_predpinn_{}.txt'.format(case),d_p.flatten().tolist())
#parameters
np.savetxt(out +'beta_predpinn_{}.txt'.format(case),beta_p.flatten().tolist())
np.savetxt(out +'gamma_predpinn_{}.txt'.format(case),gamma_p.flatten().tolist())
np.savetxt(out +'delta_predpinn_{}.txt'.format(case),delta_p.flatten().tolist())
np.savetxt(out +'compliance_predpinn_{}.txt'.format(case),compliance_p.flatten().tolist())
#loss
np.savetxt(out +'lossDatapinn_{}.txt'.format(case),model.loss_data)
np.savetxt(out +'lossPhypinn_{}.txt'.format(case),model.loss_phys)
np.savetxt(out +'total_losspinn_{}.txt'.format(case),model.total_loss)


##Get errors
I_or =I_original.reshape(-1,)
I_p  =i_p.reshape(-1,)
test_actual=I_or*N0
test_pred =I_p*N0
rmse =np.sqrt(mean_squared_error(test_actual, test_pred))
mape =np.linalg.norm((test_pred-test_actual),2)/np.linalg.norm(test_actual, 2)
ev =1- (np.var(test_pred-test_actual)/np.var(test_actual))
rel =np.sum((test_actual-test_pred)**2/(test_actual**2))
#Error Metrics
with open(out+'TNpinn.txt', 'w') as f:
    print("Error metrics ", file=f)
    print("=======================================================\n", file=f)
    print('Total CPU Time',formatted_time , file=f)
    print('RMSE',rmse, file=f)
    print('MAPE',mape, file=f)
    print('EV', ev, file=f)
    print('REL', rel, file=f)
    print("=======================================================\n", file=f)