import numpy as np
import pandas as pd
import os
import datetime
# from pyDOE import lhs
import sys
sys.path.insert(0, '../../Utilities/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
from collections import OrderedDict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score
import timeit
from tqdm import tqdm
#Set default dtype to float32
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

##Output Folders
c_dir =os.getcwd()
path = '/FIGURES/'
out = c_dir +path
if not os.path.exists(out):
    os.makedirs(out)



class BasePINN(nn.Module):
    def __init__(self, input_dim):
        super(BasePINN, self).__init__()
        # self.l_b = l_b
        # self.u_b = u_b
        self.attention = nn.MultiheadAttention(1, num_heads=1)
        self.fc1 = nn.Linear(input_dim, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 1)

    def forward(self, t):
        # if torch.is_tensor(t) != True:
        #     t = torch.from_numpy(t)
        # l_b = torch.tensor(self.l_b).float()
        # u_b = torch.tensor(self.u_b).float()
        # t = 2 * ((t - l_b) / (u_b - l_b)) - 1.0
        x = t.float()
        x = x.unsqueeze(1)  # Add an extra dimension
        x = x.permute(2, 0, 1)  # Reshape to (seq_len, batch_size, input_dim, 1)
        x = self.attention(x, x, x)[0]
        x = x.squeeze(dim=0)
        x = x.permute(1, 0, 2)  # Reshape to (seq_len, batch_size, input_dim)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = x.squeeze(dim=2)  # Remove the third dimension (input_dim)
        return x[:,0:1]  # Return (261, 1) shape

    
def loss_function(PINNs, y_true, t, eps):
    t.requires_grad=True
    S_true = y_true[:, 0:1]
    I_true = y_true[:, 1:2]
    R_true = y_true[:, 2:3]
    D_true = y_true[:, 3:4]
    S, I, R, D, beta, gamma, delta, compliance = [pinn(t) for pinn in PINNs]
    # Calculate the derivative of each compartment with respect to time
    S_t=autograd.grad(S,t,torch.ones_like(S),
                        retain_graph=True, create_graph=True)[0]
    I_t=autograd.grad(I,t,torch.ones_like(I),
                        retain_graph=True, create_graph=True)[0]
    R_t=autograd.grad(R,t,torch.ones_like(R),
                        retain_graph=True, create_graph=True)[0]
    D_t=autograd.grad(D,t,torch.ones_like(D),
                        retain_graph=True, create_graph=True)[0]

    e1 = S_t + beta * (1.0 - compliance) * S * I
    e2 = I_t - beta * (1.0 - compliance) * S * I + gamma * I + delta * I
    e3 = R_t - gamma * I
    e4 = D_t - delta * I
    e5 = 1.0 - (S + I + R + D)

    # Get the residual loss
    lossPhy = torch.mean(torch.abs(e1)) + torch.mean(torch.abs(e2)) + \
              torch.mean(torch.abs(e3)) + torch.mean(torch.abs(e4)) + \
              torch.mean(torch.abs(e5))
    S_loss =torch.mean(torch.abs(S_true - S))
    I_loss =torch.mean(torch.abs(I_true - I))
    R_loss =torch.mean(torch.abs(R_true - R))
    D_loss =torch.mean(torch.abs(D_true - D))

    lossData = (I_loss + R_loss + S_loss + D_loss)
    overall_loss = (1 - eps) * lossData + eps * lossPhy
    return overall_loss, lossPhy, lossData

def train_PINNs(PINNs, train_data, t_train, val_data,t_val, epochs, optimizer, eps):
    t_train.requires_grad=True
    trainloss=[]
    valloss =[]
    totaltime=[]
    start_time = time.time()
    for epoch in range(epochs):
        # PINNs.train()
        train_loss, _, _= loss_function(PINNs, train_data,  t_train, eps)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        trainloss.append(train_loss)
        # PINNs.eval()
        val_loss, _, _= loss_function(PINNs, val_data,  t_val, eps)
        optimizer.zero_grad()
        val_loss.backward()
        optimizer.step()
        valloss.append(val_loss)
        elapsed = time.time() - start_time
        totaltime.append(elapsed)
        
        if epoch % 1000== 0:
            elapsed = time.time() - start_time
            print(f"Epoch: {epoch}, Train Loss: {train_loss.item():.5f}, Val Loss: {val_loss.item():.5f}, Time: {elapsed:.3f} seconds")
    return trainloss, valloss, totaltime


data =pd.read_csv("tndata.csv")
def data_preprocess(data, lb, ub, N0, npoints,cs ="yes"):
    tdat=data.reindex(index=data.index[::-1])
    ic = tdat["TOTAL_CASES"]
    dc = tdat["TOTAL_DEATHS"]
    re = tdat["TOTAL_INACTIVE_RECOVERED"]
    y1, y2 =np.array(ic.values), np.array(dc.values)
    y3   =np.array(re.values)
    y4 =y1-y3-y2 #infected cases
    I, R, D =y1[lb:ub],y3[lb:ub], y2[lb:ub]
    S =N0-I-R-D
    length =int(ub-lb)
    T = np.arange(0,length).reshape(length,1)
    dd =np.arange(length)
    if cs =="yes":
        s1 =CubicSpline(dd,S)
        s2 =CubicSpline(dd,I)
        s3 =CubicSpline(dd,R)
        s4 =CubicSpline(dd,D)
        tt=np.linspace(0,length, npoints)
        x, yy1, yy2, yy3, yy4 =tt, s1(tt), s2(tt), s3(tt), s4(tt)
        TT =x.reshape((-1,1))
        SS =yy1.reshape((-1,1))
        II =yy2.reshape((-1,1))
        RR =yy3.reshape((-1,1))
        DD=yy4.reshape((-1,1))
    else:
        TT=T
        SS =S.reshape((-1,1))
        II =I.reshape((-1,1))
        RR =R.reshape((-1,1))
        DD =D.reshape((-1,1))
    return SS, II, RR, DD, TT

N0 = 6.82*1.e6
lb =19 #change this
ub =280
length=int(ub-lb)
S, I, R, D, T=data_preprocess(data, lb, ub, N0, 3000, "yes")
t=torch.from_numpy(T).float()
SS=torch.from_numpy(S).float()
II=torch.from_numpy(I).float()
RR=torch.from_numpy(R).float()
DD =torch.from_numpy(D).float()
#split data into train and test
ind =int(0.8*len(S))
S_train, S_val =SS[:ind,:], SS[ind:,:]
I_train, I_val =II[:ind,:], II[ind:,:]
R_train, R_val =RR[:ind,:], RR[ind:,:]
D_train, D_val =DD[:ind,:], DD[ind:,:]
T_train, T_val =t[:ind,:], t[ind:,:]

# l_b, u_b =t.min(0), t.max(0)
PINN_beta = BasePINN(1)
PINN_beta.to(device)
PINN_gamma = BasePINN(1)
PINN_gamma.to(device)
PINN_delta = BasePINN(1)
PINN_delta.to(device)
PINN_compliance = BasePINN(1)
PINN_compliance.to(device)
PINN_S = BasePINN(1)
PINN_S.to(device)
PINN_I= BasePINN(1)
PINN_I.to(device)
PINN_R = BasePINN(1)
PINN_R.to(device)
PINN_D= BasePINN(1)
PINN_D.to(device)



PINNs = [PINN_S,PINN_I, PINN_R, PINN_D, PINN_beta, PINN_gamma, PINN_delta, PINN_compliance]
optimizer = optim.Adam(list(PINNs[0].parameters()) + list(PINNs[1].parameters()) + 
                       list(PINNs[2].parameters()) + list(PINNs[3].parameters()) +
                       list(PINNs[4].parameters()) +list(PINNs[5].parameters()) +
                       list(PINNs[6].parameters()) +list(PINNs[7].parameters()), lr=0.001)
epochs =20000
eps =1e-1

train_data =torch.hstack((S_train/N0, I_train/N0, R_train/N0,D_train/N0)).to(device)
val_data =torch.hstack((S_val/N0, I_val/N0, R_val/N0,D_val/N0)).to(device)
t_train = T_train.to(device)
t_val =T_val.to(device)

trainl, vall, t_time=train_PINNs(PINNs, train_data, t_train, val_data,t_val, epochs, optimizer,eps)

total_time =sum(t_time)


def predict_PINNs(tt):
    t_tensor = torch.tensor(tt, dtype=torch.float32)
    predictions = [pinn(t_tensor).detach().cpu().numpy() for pinn in PINNs]
    # print(predictions.shape)
    return predictions

#predict
SS_, II_, RR_, DD_, TT=data_preprocess(data, lb, ub, N0, 261, "no")
# t_ =np.transpose(TT, (2, 0, 1))
s_p, i_p, r_p, d_p, beta_p, gamma_p, delta_p, compliance_p = predict_PINNs(TT)





print(s_p.shape)
# print(s_pp.shape)
S_original=SS_/N0
I_original=II_/N0
R_original=RR_/N0
D_original=DD_/N0
case ='c1'
np.savetxt(out +'S_truepinn_{}_{}.txt'.format(case, eps),S_original.flatten().tolist())
np.savetxt(out +'I_truepinn_{}_{}.txt'.format(case, eps),I_original.flatten().tolist())
np.savetxt(out +'R_truepinn_{}_{}.txt'.format(case, eps),R_original.flatten().tolist())
np.savetxt(out +'D_truepinn_{}_{}.txt'.format(case, eps),D_original.flatten().tolist())
np.savetxt(out +'S_predpinn_{}_{}.txt'.format(case, eps),s_p.flatten().tolist())
np.savetxt(out +'I_predpinn_{}_{}.txt'.format(case, eps),i_p.flatten().tolist())
np.savetxt(out +'R_predpinn_{}_{}.txt'.format(case, eps),r_p.flatten().tolist())
np.savetxt(out +'D_predpinn_{}_{}.txt'.format(case, eps),d_p.flatten().tolist())
#parameters
np.savetxt(out +'beta_predpinn_{}_{}.txt'.format(case, eps),beta_p.flatten().tolist())
np.savetxt(out +'gamma_predpinn_{}_{}.txt'.format(case, eps),gamma_p.flatten().tolist())
np.savetxt(out +'delta_predpinn_{}_{}.txt'.format(case, eps),delta_p.flatten().tolist())
np.savetxt(out +'compliance_predpinn_{}_{}.txt'.format(case, eps),compliance_p.flatten().tolist())
#loss
np.savetxt(out +'Trainlosspinn_{}_{}.txt'.format(case, eps),trainl)
np.savetxt(out +'Vallosspinn_{}_{}.txt'.format(case, eps),vall)


N0 = 6.82*1.e6
I_or =I_original.reshape(-1,)
I_p  =i_p.reshape(-1,)
test_actual=I_or*N0
test_pred =I_p*N0
rmse =np.sqrt(mean_squared_error(test_actual, test_pred))
mape =np.linalg.norm((test_pred-test_actual),2)/np.linalg.norm(test_actual, 2)
ev =1- (np.var(test_pred-test_actual)/np.var(test_actual))
rel =np.sum((test_actual-test_pred)**2/(test_actual**2))


#Error metrics
with open(out+'BEINNout_{}.txt'.format(eps), 'w') as f:
    print("=======================================================\n", file=f)
    print('Total CPU Time',total_time , file=f)
    print('RMSE',rmse, file=f)
    print('MAPE',mape, file=f)
    print('EV', ev, file=f)
    print('REL', rel, file=f)
    print("=======================================================\n", file=f)
