import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, '../../Utilities/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score
torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(1234)
np.random.seed(1234)
c_dir =os.getcwd()
path = '/RESULTS1/'
out = c_dir +path
if not os.path.exists(out):
    os.makedirs(out)

scaler1 =MinMaxScaler()
    

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
#         self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.lstm_out(x, h)
        out = self.fc(out[:,-1])
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden
    
    
# Define the model
class AttentionLSTM(nn.Module):
    def __init__(self, n_layers, n_hidden, in_size, out_size, drop_prob=0.2):
        super(AttentionLSTM, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.out_size= out_size
        #LSTM layer
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=drop_prob)
        self.linear = nn.Linear(n_hidden, n_hidden)
        self.attn = nn.Linear(n_hidden, 1)
        self.output_layer = nn.Linear(n_hidden, out_size)

    def forward(self, x, h):
        lstm_out, (h, c_n) = self.lstm(x, h)
        linear_out = self.linear(lstm_out)
        attn_weights = torch.softmax(self.attn(linear_out), dim=1)
        attn_applied = torch.bmm(attn_weights.transpose(1, 2), lstm_out).squeeze(1)
        output = self.output_layer(attn_applied)
        return output, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden
    

    

# ##GRU class
class GRU_model(nn.Module):
    def __init__(self,  n_layers, n_hidden, in_size, out_size, drop_prob=0.2):
        super(GRU_model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.out_size= out_size
        # GRU layers
        self.gru = nn.GRU(input_size=in_size, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=drop_prob)

        # Fully connected layer
        self.fc = nn.Linear(n_hidden, out_size)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out[:,-1])
        return out, h 
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device)
        return hidden
    
    
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
    x_data, y_data =x, y
    x_train, y_train =x[:indx],y[:indx]
    x_test, y_test =x[indx:],y[indx:]
    return x_data, y_data, x_train, y_train, x_test, y_test



## Train and Evaluate Models
def train_model(tp, loader,lr,n_hidden, epochs, n_layers, batch_size,out_size, option="GRU"):
    in_size= next(iter(loader))[0].shape[2]
   
    # Instantiating the models
    if (option == "GRU"):
        model = GRU_model( n_layers, n_hidden, in_size, out_size)
    else:
        model = LSTM_model(n_layers, n_hidden, in_size, out_size)
    model.to(device)
    
   
    
    #Get the loss function
    loss_func = torch.nn.MSELoss()   
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler= torch.optim.lr_scheduler.CyclicLR(optim, base_lr=1e-5, max_lr=1e-3, step_size_up=100, 
    #                                                   mode="exp_range", gamma=0.95, cycle_momentum=False)
    
    model.train()
    # print("{} Training".format(option))
    total_time=[]
    epc_arr=[]
    loss_arr=[]
    # Start training loop
    for epoch in range(1,epochs+1):
        #Get the hidden state
        st= time.time()
        loss_avg = 0.
        c= 0
        h = model.init_hidden(batch_size)
        for x, label in loader:
            c += 1
            if (option == "GRU"):
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            
            out, h = model(x.to(device).float(), h)
            loss= loss_func(out, label.to(device).float())
            loss.backward()
            optim.step()
            # scheduler.step()
            loss_avg += loss.item()     
        ct= time.time()
        elapsed =ct-st
        epc_arr.append(epoch)
        loss_arr.append(loss_avg)
        total_time.append(elapsed)
        if epoch%100==0:
            print("Epoch %d/%d, %s Loss: %.3e, Time:%.2f seconds"%(epoch, epochs,tp, loss_avg/len(loader), elapsed))
    print('{} Total {} Time in seconds {}'.format(option,tp, str(sum(total_time))))
    return model, np.array(epc_arr), np.array(loss_arr)

def train_model_attn(tp,loader,lr,n_hidden, epochs, n_layers, batch_size,out_size, option="AGRU"):
    in_size= next(iter(loader))[0].shape[2]
   
    # Instantiating the models
    if (option == "AGRU"):
        model = AttentionGRU( n_layers, n_hidden, in_size, out_size)
    else:
        model = AttentionLSTM(n_layers, n_hidden, in_size, out_size)
    model.to(device)
    
   
    
    #Get the loss function
    loss_func = torch.nn.MSELoss()   
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler= torch.optim.lr_scheduler.CyclicLR(optim, base_lr=1e-5, max_lr=1e-3, step_size_up=100, 
    #                                                   mode="exp_range", gamma=0.95, cycle_momentum=False)
    
    model.train()
    # print("{} Training".format(option))
    total_time=[]
    epc_arr=[]
    loss_arr=[]
    # Start training loop
    for epoch in range(1,epochs+1):
        #Get the hidden state
        st= time.time()
        loss_avg = 0.
        c= 0
        h = model.init_hidden(batch_size)
        for x, label in loader:
            c += 1
            if (option == "AGRU"):
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            
            out, _= model(x.to(device).float(), h)
            loss= loss_func(out, label.to(device).float())
            loss.backward()
            optim.step()
             # scheduler.step()
            loss_avg += loss.item()     
        ct= time.time()
        elapsed =ct-st
        epc_arr.append(epoch)
        loss_arr.append(loss_avg)
        total_time.append(elapsed)
        if epoch%100==0:
            print("Epoch %d/%d, %s Loss: %.3e, Time:%.2f seconds"%(epoch, epochs,tp, loss_avg/len(loader), elapsed))
    print('{} Total {} Time in seconds {}'.format(option,tp, str(sum(total_time))))
    return model, np.array(epc_arr), np.array(loss_arr)



def evaluate_model(option, case, model1,model2, x_test, y_test, x_data, y_data, x_train, y_train,scaler1):
    model1.eval()
    start_time = time.time()
    inputs = torch.from_numpy(np.array(x_data))
    labs = torch.from_numpy(np.array(y_data))
    h = model1.init_hidden(inputs.shape[0])
    out, h = model1(inputs.to(device).float(),h)
    output=out.detach().cpu().numpy().reshape((-1,1))
    labs.numpy().reshape((-1,1))
    actual =scaler1.inverse_transform(np.array(labs))
    predicted=scaler1.inverse_transform(np.array(output))
    predicted =np.abs(predicted)
    ##Get training
    inputs_train =torch.from_numpy(np.array(x_train))
    # labs1 = torch.from_numpy(np.array(y_train))
    h = model1.init_hidden(inputs_train.shape[0])
    out, h = model1(inputs_train.to(device).float(), h)
    output1=out.detach().cpu().numpy().reshape((-1,1))
    train_actual =scaler1.inverse_transform(np.array(y_train))
    train_pred =scaler1.inverse_transform(np.array(output1))
    
    ##Get Testing
    model1.eval()
    inputs_test =torch.from_numpy(np.array(x_test))
    # labs2 = torch.from_numpy(np.array(y_test))
    h = model1.init_hidden(inputs_test.shape[0])
    out, h = model1(inputs_test.to(device).float(), h)
    output2=out.detach().cpu().numpy().reshape((-1,1))
    test_actual =scaler1.inverse_transform(np.array(y_test))
    test_pred =scaler1.inverse_transform(np.array(output2))
    test_pred =np.abs(test_pred)
    print("Evaluation Time: {}".format(str(time.time()-start_time)))
    
    ##Get errors
    rmse =np.sqrt(mean_squared_error(test_actual, test_pred))
    mape =np.linalg.norm((test_pred-test_actual),2)/np.linalg.norm(test_actual, 2)
    ev =1- (np.var(test_pred-test_actual)/np.var(test_actual))
    rel =np.sum((test_actual-test_pred)**2/(test_actual**2))
    
    print("Error metrics for {} for case {}".format(option, case))
    print("=======================================================\n")
    print('RMSE',rmse)
    print('MAPE',mape)
    print('EV', ev)
    print('REL', rel)
    print("=======================================================\n")
    return actual,predicted, train_actual, train_pred, test_actual, train_pred


    
## Run models
torch.manual_seed(0)
np.random.seed(1234)
lr= 0.01
# in_size = 1
n_hidden = 16 #when neurons =16, 32
n_layers = 2 #when layers =2,3, 4, 5, 7
out_size = 1
n, l =16, 2
def run_model(data, option, cs, epochs, batch_size, case):
    print('{} outcomes............'.format(cs))
    x_data, y_data, x_train, y_train, x_test, y_test=split_data(data,4, scaler1, 0.8)
    #Training               
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    model1, ep1, loss1 = train_model("Train",train_loader,lr,n_hidden, epochs, n_layers, batch_size, out_size, option=option)
    #Testing               
    test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    model2, ep2, loss2 = train_model("Test", test_loader,lr,n_hidden, epochs, n_layers, batch_size, out_size, option=option)              
    y_true, y_pred, tr_a, tr_p, ts_a, ts_p =evaluate_model(option, case, model1,model2, x_test, y_test, x_data, y_data, x_train, y_train,scaler1)
    return y_true, y_pred, tr_a, tr_p, ts_a, ts_p, ep1, loss1, loss2

def future_prediction(data, option, cs, epochs, batch_size, n_steps):
    print('{} outcomes............'.format(cs))
    x_data, y_data, x_train, y_train, x_test, y_test=split_data(data,200, scaler1, 0.8)
    #Training               
    train_data = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    model, ep, loss = train_model("Train",train_loader,lr,n_hidden, epochs, n_layers, batch_size, out_size, option=option)
    model.eval()
    km =len(data)
    x_input=y_data[km-200-n_steps:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    i=0
    while(i<10):
        if(len(temp_input)>n_steps):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            x_input = torch.from_numpy(np.array(x_input))
            h = model.init_hidden(x_input.shape[0])
            out1, h = model(x_input.to(device).float(), h)
            yhat=out1.detach().cpu().numpy().reshape((-1,1))
            
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            x_input = torch.from_numpy(np.array(x_input))
            h = model.init_hidden(x_input.shape[0])
            out1, h = model(x_input.to(device).float(), h)
            yhat=out1.detach().cpu().numpy().reshape((-1,1))
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    return np.array(lst_output)



def run_model_attn(data, option, cs, epochs, batch_size, case):
    print('{} outcomes............'.format(cs))
    x_data, y_data, x_train, y_train, x_test, y_test=split_data(data,4, scaler1, 0.8)
    #Training               
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    model1, ep1, loss1 = train_model_attn("Train", train_loader,lr,n_hidden, epochs, n_layers, batch_size, out_size, option=option)
    #Testing               
    test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    model2, ep2, loss2 = train_model_attn("Test", test_loader,lr,n_hidden, epochs, n_layers, batch_size, out_size, option=option)              
    y_true, y_pred, tr_a, tr_p, ts_a, ts_p =evaluate_model(option, case,model1,model2, x_test, y_test, x_data, y_data, x_train, y_train,scaler1)
    return y_true, y_pred, tr_a, tr_p, ts_a, ts_p, ep1, loss1, loss2



def future_prediction_attn(data, option, cs, epochs, batch_size, n_steps):
    print('{} outcomes............'.format(cs))
    x_data, y_data, x_train, y_train, x_test, y_test=split_data(data,200, scaler1, 0.8)
    #Training               
    train_data = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    model, ep, loss = train_model_attn("Train",train_loader,lr,n_hidden, epochs, n_layers, batch_size, out_size, option=option)
    model.eval()
    km =len(data)
    x_input=y_data[km-200-n_steps:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    i=0
    while(i<10):
        if(len(temp_input)>n_steps):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            x_input = torch.from_numpy(np.array(x_input))
            h = model.init_hidden(x_input.shape[0])
            out1, h = model(x_input.to(device).float(), h)
            yhat=out1.detach().cpu().numpy().reshape((-1,1))
            
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            x_input = torch.from_numpy(np.array(x_input))
            h = model.init_hidden(x_input.shape[0])
            out1, h = model(x_input.to(device).float(), h)
            yhat=out1.detach().cpu().numpy().reshape((-1,1))
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    return np.array(lst_output)


#LSTM


data_rec =np.load("Figures/data.npy")
I =data_rec[:,0:1]
C =data_rec[:,1:2]
T =data_rec[:,2:3]
Reff =data_rec[:,3:4]
_, yp1, tr_a, tr_p, ts_a, ts_p, ep, l11, l21 =run_model(I, "LSTM", "TN", 1500, 20, "inf") #infected
_, yp2, tr_a, tr_p, ts_a, ts_p, ep, l12, l22 =run_model(C, "LSTM", "TN", 1500, 20, "com") #compliance
_, yp3, tr_a, tr_p, ts_a, ts_p, ep, l13, l23 =run_model(T, "LSTM", "TN", 1500, 20, "tran") #transmission
_, yp4, tr_a, tr_p, ts_a, ts_p, ep, l14, l24 =run_model(Reff, "LSTM", "TN", 1500, 20, "reff") #reff

##save the outcomes
np.savetxt(out+"ylstm1.txt", yp1)
np.savetxt(out+"ylstm2.txt", yp2)
np.savetxt(out+"ylstm3.txt", yp3)
np.savetxt(out+"ylstm4.txt", yp4)
#train loss
np.savetxt(out+"lstm_train_loss1.txt", l11)
np.savetxt(out+"lstm_train_loss2.txt", l12)
np.savetxt(out+"lstm_train_loss3.txt", l13)
np.savetxt(out+"lstm_train_loss4.txt", l14)

#test loss
np.savetxt(out+"lstm_test_loss1.txt", l21)
np.savetxt(out+"lstm_test_loss2.txt", l22)
np.savetxt(out+"lstm_test_loss3.txt", l23)
np.savetxt(out+"lstm_test_loss4.txt", l24)



#A-LSTM
_, yp1a, tr_a, tr_p, ts_a, ts_p, ep, l11a, l21a =run_model_attn(I, "ALSTM", "TN", 1500, 20, "inf") #infected
_, yp2a, tr_a, tr_p, ts_a, ts_p, ep, l12a, l22a =run_model_attn(C, "ALSTM", "TN", 1500, 20, "com") #compliance
_, yp3a, tr_a, tr_p, ts_a, ts_p, ep, l13a, l23a =run_model_attn(T, "ALSTM", "TN", 1500, 20, "tran") #transmission
_, yp4a, tr_a, tr_p, ts_a, ts_p, ep, l14a, l24a =run_model_attn(Reff, "ALSTM", "TN", 1500, 20, "reff") #reff

#save the outcomes
np.savetxt(out+"yalstm1.txt", yp1a)
np.savetxt(out+"yalstm2.txt", yp2a)
np.savetxt(out+"yalstm3.txt", yp3a)
np.savetxt(out+"yalstm4.txt", yp4a)
#train loss
np.savetxt(out+"alstm_train_loss1.txt", l11a)
np.savetxt(out+"alstm_train_loss2.txt", l12a)
np.savetxt(out+"alstm_train_loss3.txt", l13a)
np.savetxt(out+"alstm_train_loss4.txt", l14a)

#test loss
np.savetxt(out+"alstm_test_loss1.txt", l21a)
np.savetxt(out+"alstm_test_loss2.txt", l22a)
np.savetxt(out+"alstm_test_loss3.txt", l23a)
np.savetxt(out+"alstm_test_loss4.txt", l24a)



