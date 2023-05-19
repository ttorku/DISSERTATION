import numpy as np
import pandas as pd
import os
import datetime
from pyDOE import lhs
import sys
sys.path.insert(0, '../../Utilities/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
from collections import OrderedDict
import timeit
from tqdm import tqdm
#Set default dtype to float32
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)
#############################Get Data################
dat =pd.read_csv("tndata.csv")
scaler =MinMaxScaler()

def data_preprocess(dat, npoints, scaler, cs ="yes"):
    tdat=dat.reindex(index=dat.index[::-1])
    Inew=tdat['TOTAL_CONFIRMED']
    Rnew =tdat['TOTAL_INACTIVE_RECOVERED']
    nn =len(Rnew)
    N =100.00
    dd =np.arange(nn)
    if cs =="yes":
        s1 =CubicSpline(dd,Inew[:nn])
        s2 =CubicSpline(dd,Rnew[:nn])
        tt=np.linspace(0,nn, npoints)
        x, y1, y2 =tt, s1(tt), s2(tt)
        t, I1, R1 =x.reshape((-1,1)), y1.reshape((-1,1)), y2.reshape((-1,1))
        I2 =I1*100/(7.05*10**6)
        R2 =R1*100/(7.05*10**6)
        S2 =N-I2-R2
    else:
        tt=np.linspace(0,nn, npoints)
        y1 =np.array(Inew[:nn])
        y2 =np.array(Rnew[:nn])
        t, I1, R1 =tt.reshape((-1,1)), y1.reshape((-1,1)), y2.reshape((-1,1))
        I2 =I1*100/(7.05*10**6)
        R2 =R1*100/(7.05*10**6)
        S2 =N-I2-R2    
    return t, S2, I2, R2


class DNN(torch.nn.Module):
    def __init__(self, layers,lb, ub):
        super(DNN, self).__init__()
        self.activation = nn.Tanh()
        self.layers =layers
        self.lb, self.ub =lb, ub
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]) 
        for i in range(len(self.layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data) 
            
    def forward(self,t):
        if torch.is_tensor(t) != True:         
            t = torch.from_numpy(t)
        l_b, u_b =self.lb, self.ub
        t = 2*((t - l_b)/(u_b - l_b))-1.0
        a = t.float()
        for i in range(len(layers)-2):  
            z = self.linears[i](a)
            a = self.activation(z)  
        a = self.linears[-1](a)
        return a
    
# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self,  layers, S,I, R, t, lb, ub, v,eta, N, lr, b, g):
        
        # boundary conditions
        self.lb = torch.tensor(lb).float()
        self.ub = torch.tensor(ub).float()
        self.v, self.eta, self.N =v, eta, N
        self.layers =layers
        
        # data
        self.t = t
        self.S = S
        self.R = R
        self.I = I
        self.log_loss =[]
        
        # settings
        self.beta= torch.tensor([b], requires_grad=True)
        self.gamma= torch.tensor([g], requires_grad=True)
        
        self.beta= torch.nn.Parameter(self.beta)
        self.gamma= torch.nn.Parameter(self.gamma)
        
        # deep neural networks
        self.dnn = DNN(self.layers, self.lb, self.ub)
        self.dnn.register_parameter('beta', self.beta)
        self.dnn.register_parameter('gamma', self.gamma)
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=lr)
        self.iter = 0
        

          
    def net_sir(self,t):
        N=self.N
        out= self.dnn(t)
        I, R =out[:,0:1], out[:,1:2]
        S =N-I-R
        return S,I, R
    
    def net_residual(self, t):
        beta, gamma =self.beta, self.gamma
        t.requires_grad=True 
        S, I, R= self.net_sir(t)
        S_t=autograd.grad(S,t,torch.ones_like(S),
                        retain_graph=True, create_graph=True)[0]
        I_t=autograd.grad(I,t,torch.ones_like(I),
                        retain_graph=True, create_graph=True)[0]
        R_t=autograd.grad(R,t,torch.ones_like(R),
                        retain_graph=True, create_graph=True)[0]
        v, eta, N=self.v, self.eta, self.N
        e1 =S_t +(beta*S*I)/N +v*eta*S/N
        e2 =I_t -(beta*S*I)/N +gamma*I
        e3 =R_t -gamma*I-v*eta*S/N
        return e1,e2,e3

    def loss_func(self):
        S_pred, I_pred, R_pred = self.net_sir(self.t)
        e1, e2, e3= self.net_residual(self.t)
        loss =torch.mean(torch.square(self.I - I_pred)) +\
                torch.mean(torch.square(self.S - S_pred)) +\
                torch.mean(torch.square(self.R - R_pred)) +\
                torch.mean(torch.square(self.I[0] - I_pred[0])) +\
                torch.mean(torch.square(self.R[0] - R_pred[0])) +\
                torch.mean(torch.square(self.S[0] - S_pred[0])) +\
                0.01*torch.mean(torch.square(e1)) +\
                 0.01*torch.mean(torch.square(e2)) +\
                 0.01*torch.mean(torch.square(e3))
        return loss
    
    def train(self, nIter):
        self.dnn.train()
        self.log_loss =[]
        for epoch in tqdm(range(nIter)):
            loss =self.loss_func()
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            self.log_loss.append(loss.item())
            if epoch % 100 == 0:
                print('Loss: %e, beta: %.5f, gamma: %.5f' % (loss.item(), self.beta.item(), self.gamma.item()))
    
    def predict(self, t):
        self.dnn.eval()
        Sh, Ih, Rh= self.net_sir(t)
        Sh = Sh.detach().numpy()
        Ih = Ih.detach().numpy()
        Rh = Rh.detach().numpy()
        return Sh,Ih, Rh
    
    
    
c_dir =os.getcwd()
path = '/Figures/'
out = c_dir +path
if not os.path.exists(out):
    os.makedirs(out)
    
def transfer_sde(data,sigma2):
    m =len(data)
    y_inf =np.zeros((m,1))
    for j in range(m):
        y_inf[j,:] =data[j,:]+np.random.normal(loc=0.0, scale=sigma2, size=(1,1))
    df1 =pd.DataFrame(y_inf)
    df1_ar =df1.rolling(7).mean()
    a_ar=np.array(df1_ar[6:]).reshape(-1,1)
    a_sm=np.vstack((data[:6,:], a_ar))
    return y_inf

N =100.0
npoints =len(dat)
eta =float(sys.argv[1])
v =float(sys.argv[2])
epochs =int(sys.argv[3])
lr=float(sys.argv[4])
neu=int(sys.argv[5])
b =float(sys.argv[6])
g =float(sys.argv[7])
n =int(sys.argv[8])
sigma2 =float(sys.argv[9])
t1, S1, I1, R1 =data_preprocess(dat,  npoints,  "no")
S2 =transfer_sde(S1,sigma2)
I2 =transfer_sde(I1, sigma2)
R2 =transfer_sde(R1, sigma2)
layers =[1] +7*[neu] +[3]
S_list =[]
I_list =[]
R_list =[]
beta_list=[]
gamma_list =[]
log_loss =[]

for k in range(n):
    print("{} Run ........".format(k))
    lb, ub =t1.min(0), t1.max(0)
    t = torch.from_numpy(t1)
    I =torch.from_numpy(I2)
    R =torch.from_numpy(R2)
    S =torch.from_numpy(S2)
    t=t.float()
    I=I.float()
    R=R.float()
    S=S.float()
    # training
    model = PhysicsInformedNN( layers, S,I, R, t, lb, ub, v,eta, N, lr, b, g)
    model.train(epochs)  #for Adam
    S_pred, I_pred, R_pred = model.predict(t1)
    beta_value = model.beta.detach().numpy()
    gamma_value = model.gamma.detach().numpy() 
    beta_v =np.round(beta_value, 5)
    gamma_v =np.round(gamma_value, 5)
    print('beta: %.5f'% (beta_value))                             
    print('gamma: %.5f'% (gamma_value))
    print('R0: %.5f'%(beta_v/gamma_v))
    S_list.append(S_pred.flatten().tolist())
    I_list.append(I_pred.flatten().tolist())
    R_list.append(R_pred.flatten().tolist())
    beta_list.append(beta_v)
    gamma_list.append(gamma_v)
    log_loss.append(model.log_loss)
    
np.savetxt(out +'S_{}_{}.txt'.format(eta, v), S_list)
np.savetxt(out +'I_{}_{}.txt'.format(eta, v), I_list)
np.savetxt(out +'R_{}_{}.txt'.format(eta, v), R_list)
np.savetxt(out +'beta_{}_{}.txt'.format(eta, v),  beta_list)
np.savetxt(out +'gamma_{}_{}.txt'.format(eta, v),gamma_list)
np.savetxt(out +'train_loss_{}_{}.txt'.format(eta, v),log_loss)
