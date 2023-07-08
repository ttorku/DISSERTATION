import numpy as np
import time 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
import sys
sys.path.insert(0, '../../Utilities/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.dates as dates
torch.manual_seed(125)
if is_cuda :
    torch.cuda.manual_seed_all(125)

scaler1=MinMaxScaler()
lookback =4
def data_lstm(data, lookback, scaler1):
    """
    Input: data and time steps
    """
    ndt=scaler1.fit_transform(data)
#     ndt =data
    x_ar =[]
    y_ar =[]
    n =len(data)
    for k in range(n):
        ini =k+lookback
        if (ini)>n-1:
            break
        xs, ys =ndt[k:ini], ndt[ini]
        x_ar.append(xs)
        y_ar.append(ys)
        x, y =np.array(x_ar), np.array(y_ar) 
    
    return x,y
def split_data(data,lookback, scaler1, split):
    x, y = data_lstm(data, lookback, scaler1)
    indx =int(split*len(y))
    ##convert data to torch
    x_data = Variable(torch.Tensor(np.array(x)))
    y_data = Variable(torch.Tensor(np.array(y)))
    x_train= Variable(torch.Tensor(np.array(x[:indx])))
    y_train= Variable(torch.Tensor(np.array(y[:indx])))
    x_test= Variable(torch.Tensor(np.array(x[indx:])))
    y_test= Variable(torch.Tensor(np.array(y[indx:])))
    return x_data, y_data, x_train, y_train, x_test, y_test

##LSTM class
class LSTM_model(nn.Module):

    def __init__(self, n_layers, n_hidden, in_size, out_size, drop_prob=0.2):
        super(LSTM_model, self).__init__()
        
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.out_size= out_size
        #LSTM layer
        self.lstm_out = nn.LSTM(input_size=in_size, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=drop_prob)
            
        ###Fully connected layer
        self.fc = nn.Linear(n_hidden, out_size)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.n_hidden))
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.n_hidden))
        
        ula, (h_out, _) = self.lstm_out(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.n_hidden)
        
        out = self.fc(h_out)
        
        return out
class BiLSTM_model(nn.Module):

    def __init__(self, n_layers, n_hidden, in_size, out_size, drop_prob=0.2):
        super(BiLSTM_model, self).__init__()
        
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.out_size= out_size
        #LSTM layer
        self.bilstm = nn.LSTM(input_size=in_size, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, bidirectional=True, dropout=drop_prob)
            
        ###Fully connected layer
        self.fc = nn.Linear(n_hidden*2, out_size)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers*2, x.size(0), self.n_hidden))
        c_0 = Variable(torch.zeros(self.n_layers*2, x.size(0), self.n_hidden))
        out, _= self.bilstm(x, (h_0, c_0))
        out = self.fc(out[:,-1,:])
        return out 
##GRU class
class GRU_model(nn.Module):
    def __init__(self,  n_layers, n_hidden, in_size, out_size, drop_prob=0.2):
        super(GRU_model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.out_size= out_size
#         self.dp =dp

        # GRU layers
        self.gru = nn.GRU(input_size=in_size, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=drop_prob)

        # Fully connected layer
        self.fc = nn.Linear(n_hidden, out_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).requires_grad_()
        out, _ = self.gru(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out    

def train_models(n_layers, n_hidden, in_size,out_size, option, lr, epochs, trainX, trainY):
    model_out = option(n_layers, n_hidden, in_size, out_size)
    #Get the loss function
    loss_func = torch.nn.MSELoss()   
    optim = torch.optim.Adam(model_out.parameters(), lr=lr)
    epc_arr=[]
    loss_arr=[]
    total_time=[]
    # Train the model
    for epoch in range(epochs+1):
        st= time.time()
        outputs =  model_out(trainX)
        optim.zero_grad()
        # obtain the loss function
        loss = loss_func(outputs, trainY)
        loss.backward()
        optim.step()
        if epoch % 500 == 0:
            print("Epoch: %d, loss: %.3e" % (epoch, loss.item()))
        ct= time.time()
        elapsed =ct-st
        epc_arr.append(epoch)
        loss_arr.append(loss.item())
        total_time.append(elapsed)
    print('{} Total Training Time in seconds {}'.format(option, str(sum(total_time))))
    return model_out, np.array(epc_arr), np.array(loss_arr)

from sklearn.model_selection import train_test_split
torch.manual_seed(0)
np.random.seed(1234)
batch_size = 100
epochs = 2000
lr= 0.01
in_size = 1
n_hidden = 16 #when neurons =16, 32
n_layers = 1  #when layers =2,3, 4, 5, 7
out_size = 1
# n, l =24, 1

def split_data1(data,lookback, scaler1, split):
    x, y = data_lstm(data, lookback, scaler1)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=split, random_state=42)
    ##convert data to torch
    x_data = Variable(torch.Tensor(np.array(x)))
    y_data = Variable(torch.Tensor(np.array(y)))
    x_train= Variable(torch.Tensor(X_train))
    y_train= Variable(torch.Tensor(Y_train))
    x_test= Variable(torch.Tensor(X_test))
    y_test= Variable(torch.Tensor(Y_test))
    return x_data, y_data, x_train, y_train, x_test, y_test
def cross_validation(k, data, scaler1, split, option, cs, name):
    x_data, y_data, x_train, y_train, x_test, y_test=split_data1(data,4, scaler1, 0.8)
    num_val =len(y_train)//k
    all_score =[]
    all_mse_loss=[]
    total_time=[]
    t0 =time.time()
    for j in range(k):
        print(j)
        val_x =x_train[j*num_val:(j+1)*num_val,:, :]
        val_y =y_train[j*num_val:(j+1)*num_val]
        ##Do the k-fold
        partial_x =torch.cat([x_train[:j*num_val,:,:], x_train[(j+1)*num_val:,:,:]], axis=0)
        partial_y =torch.cat([y_train[:j*num_val], y_train[(j+1)*num_val:]], axis=0)
        print('####### {} bootstrap for {} #######'.format(name,cs))
        model, ep, loss = train_models(n_layers, n_hidden, in_size, out_size, option,lr, epochs, partial_x,partial_y)
        model.eval()
        yhat = model(val_x)
        yhat = yhat.data.numpy()
        val_data =val_y.data.numpy()
        elapsed =time.time()-t0
        total_time.append(elapsed)
#         print('CPU Time: %.2f'%(elapsed))
        score =np.sqrt(mean_squared_error(yhat, val_data))
        all_score.append(score)
        all_mse_loss.append(loss)
    val =np.array(all_mse_loss)
    ym =np.mean(val, axis=0)
    print('Average RMSE score {} for {}'.format(np.mean(all_score), name) )
    print('std score {} for {}'.format(np.std(all_score),name) )
    print('Total Cross Val Time %.2f'%(sum( total_time)))
    return ym, ep, np.mean(all_score), sum( total_time)

def plot_cross_val(k, out, data1, cs):
    l_means1=[]
    b_means1=[]
    g_means1 =[]
    l_cpu1 =[]
    b_cpu1 =[]
    g_cpu1 =[]
    for i in range(len(k)):
        print(i)
        _,_, m_l1, cpu_l1 = cross_validation(k[i], data1, scaler1, 0.8, LSTM_model, cs, "ResNet-LSTM")
        _,_, m_b1, cpu_b1 = cross_validation(k[i], data1, scaler1, 0.8, BiLSTM_model, cs, "ResNet-BiLSTM")
        _,_, m_g1, cpu_g1= cross_validation(k[i], data1, scaler1, 0.8, GRU_model, cs, "ResNet-GRU")
        l_means1.append(m_l1)
        b_means1.append(m_b1)
        g_means1.append(m_g1)
        l_cpu1.append(cpu_l1)
        b_cpu1.append(cpu_b1)
        g_cpu1.append(cpu_g1)
    kv =np.array(k)
    l_cpu1 =np.array(l_cpu1)
    b_cpu1 =np.array(b_cpu1)
    g_cpu1 =np.array(g_cpu1)
    font = 24
    ind = np.arange(len(l_means1))  # the x locations for the groups
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects4 = ax.bar(ind - width, tuple(l_means1), width, 
                label='ResNet-LSTM', align='center')
    rects5 = ax.bar(ind , tuple(b_means1), width, 
                label='ResNet-BiLSTM', align='center')
    rects6 = ax.bar(ind +width, tuple(g_means1), width, 
                label='ResNet-GRU', align='center')
    ax.legend(fontsize=22)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    ax.set_xticks(ind)
    ax.set_xticklabels((r'$k=4$', r'$k=5$', r'$k=6$', r'$k=7$'))
    ax.tick_params(axis='both', labelsize = 24)
    ax.set_xlabel('Values of K', fontsize = font)
    ax.set_ylabel('Average RMSE Scores', fontsize = font)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_title('Cross Validation Scores',  fontsize = font)
    fig.set_size_inches(w=13,h=6.5)
    plt.savefig(out+'crossresnetBar{}.png'.format(cs))
#     plt.show()
    ax.autoscale(tight=True)
    plt.show()
    return