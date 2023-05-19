import numpy as np
np.seterr(all='warn')
import pandas as pd
import math
import torchsde
import os
import datetime
from pyDOE import lhs
import sys
sys.path.insert(0, '../../Utilities/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
from torch.utils.data import TensorDataset, DataLoader
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
from scipy.special import logsumexp
#Set default dtype to float32
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

#torch
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
class PhysicsInformedNN(torch.nn.Module):
    def __init__(self, layers,  S, I, R, t, lb, ub, v,eta,N, lr, b, g, M, sigma2):
        super(PhysicsInformedNN, self).__init__()
        # boundary conditions
        self.lb = torch.tensor(lb).float()
        self.ub = torch.tensor(ub).float()
        self.v, self.eta, self.N=v, eta, N
        self.layers =layers
        self.M =M
        self.sigma2 =sigma2
        # data
        self.t = t
        self.R = R
        self.I = I
        self.S =S
        # deep neural networks
        # settings
        self.beta= torch.tensor([b], requires_grad=True)
        self.gamma= torch.tensor([g], requires_grad=True)
        
        self.beta= torch.nn.Parameter(self.beta)
        self.gamma= torch.nn.Parameter(self.gamma)
        
        # deep neural networks
        self.dnn = DNN(self.layers, self.lb, self.ub)
        self.dnn.register_parameter('beta', self.beta)
        self.dnn.register_parameter('gamma', self.gamma)
        self.loss= []
        self.save = 3
        
#         #optimizer
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr =lr, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=100, 
                                                      mode="exp_range", gamma=0.85, cycle_momentum=False)
#         self.iter = 0
      
    def net_sir(self,t):
        N=self.N
        out= self.dnn(self.t)
        I, R =out[:,0:1], out[:,1:2]
        S =N-I-R
        return S,I, R 
    
    def net_stochastic_mcmc(self,t):
        S,I, R  =self.net_sir(t)
        beta, gamma =self.beta, self.gamma
        eta=self.eta
        v =self.v
        N=self.N
        M =self.M
        dt = torch.Tensor([1.0]) 
        sigma2 =self.sigma2
        s_list =[]
        i_list =[]
        r_list =[]
        #MCMC simulation
        for i in range(M):
            S_k =S +dt*(-beta*I*S/N -v*eta*S/N)+ sigma2*S*math.sqrt(dt) * torch.rand(t.size())
            I_k =I + dt*(beta*I*S/N -gamma*I) + sigma2*I*math.sqrt(dt) * torch.rand(t.size())
            R_k =R +dt* (gamma*I +v*eta*S/N) +sigma2*R*math.sqrt(dt) * torch.rand(t.size())
            s_list.append(S_k)
            i_list.append(I_k)
            r_list.append(R_k)
        s_mean =torch.mean(torch.stack(s_list),0)
        i_mean =torch.mean(torch.stack(i_list),0)
        r_mean =torch.mean(torch.stack(r_list),0)
        return s_mean, i_mean, r_mean

    def loss_func(self):
        S_pred,I_pred, R_pred  =self.net_sir(t)
        S_k, I_k, R_k =self.net_stochastic_mcmc(t)
        loss =torch.mean(torch.square(self.I - I_pred)) +\
               torch.mean(torch.square(self.S - S_pred)) +\
               torch.mean(torch.square(self.R - R_pred)) +\
                torch.mean(torch.square(self.I[0] - I_pred[0])) +\
                 torch.mean(torch.square(self.S[0] - S_pred[0])) +\
                 torch.mean(torch.square(self.R[0] - R_pred[0]))+\
                0.00001*torch.mean(torch.square(R_k-R_pred)) +\
                0.00001*torch.mean(torch.square(S_k-S_pred))+\
                0.00001*torch.mean(torch.square(I_k-I_pred))
        return loss
    
    def train(self, nIter):
        start_time = timeit.default_timer()
        self.dnn.train()
        self.loss =[]
        for epoch in tqdm(range(nIter)):
            los =self.loss_func()
            loss =los 
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.loss.append(loss.item())
            if epoch % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                print('Loss: %e, beta: %.5f, gamma: %.5f, Time: %.2f' % (loss.item(), self.beta.item(), self.gamma.item(),elapsed))
                start_time = timeit.default_timer()
                            
    def predict(self, t):
        self.dnn.eval()
        Sh, Ih, Rh= self.net_sir(t)
        Sh = Sh.detach().numpy()
        Ih = Ih.detach().numpy()
        Rh = Rh.detach().numpy()
        return Sh,Ih, Rh
    
c_dir =os.getcwd()
path = '/Figures1/'
out = c_dir +path
if not os.path.exists(out):
    os.makedirs(out)

dat =pd.read_csv("tndata.csv")
def data_preprocess(dat, npoints, cs ="yes"):
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
sigma2 =float(sys.argv[8])
M=int(sys.argv[9])
n =int(sys.argv[10])
# decay =float(sys.argv[10])
t1, S1, I1, R1 =data_preprocess(dat,  npoints,  "no") 
S2 =transfer_sde(S1,sigma2)
I2 =transfer_sde(I1, sigma2)
R2 =transfer_sde(R1, sigma2)

layers =[1] +9*[neu] +[3]
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
    I=torch.from_numpy(I2)
    R =torch.from_numpy(R2)
    S =torch.from_numpy(S2)
    t=t.float()
    I=I.float()
    R=R.float()
    S=S.float()
    # training
    model = PhysicsInformedNN( layers,  S, I, R, t, lb, ub, v,eta,N, lr, b, g, M, sigma2)
#     model =mod.to(device)
    model.train(epochs)  #for Adam
    S_p, I_p, R_p = model.predict(t1)
    S_pred =S_p
    I_pred =I_p
    R_pred =R_p
    beta_value = model.beta.detach().numpy()
    gamma_value = model.gamma.detach().numpy() 
    beta_v =np.round(abs(beta_value), 5)
    gamma_v =np.round(abs(gamma_value), 5)
    print('beta: %.5f'% (abs(beta_value)))                             
    print('gamma: %.5f'% (abs(gamma_value)))
    print('R0: %.5f'%(beta_v/gamma_v))
    S_list.append(S_pred.flatten().tolist())
    I_list.append(I_pred.flatten().tolist())
    R_list.append(R_pred.flatten().tolist())
    beta_list.append(beta_v)
    gamma_list.append(gamma_v)
    log_loss.append(model.loss)
    
np.savetxt(out +'S_{}_{}.txt'.format(eta, v), S_list)
np.savetxt(out +'I_{}_{}.txt'.format(eta, v), I_list)
np.savetxt(out +'R_{}_{}.txt'.format(eta, v), R_list)
np.savetxt(out +'beta_{}_{}.txt'.format(eta, v),  beta_list)
np.savetxt(out +'gamma_{}_{}.txt'.format(eta, v),gamma_list)
np.savetxt(out +'train_loss_{}_{}.txt'.format(eta, v),log_loss)

PATH =out+'model.pth_{}_{}'.format(eta, v)
torch.save(model.state_dict(), PATH)
