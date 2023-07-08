### Using PINN for parameter Estimates
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tff
tff.disable_v2_behavior()
import time
start_time = time.time()
tff.set_random_seed(1234)
class sir_param:
    def __init__(self, n_layers, i,t, lb, ub,bi,gi, v, eta, tf, U0, N, split):
        ##Initialize 
        self.i, self.t, self.v, self.eta =i, t,v, eta
        self.lb, self.ub, self.N =lb, ub, N
        self.n_layers =n_layers
        self.tf =tf
        self.split =split
        indx =int(self.split*len(i))
        ##Training 
        self.s0, self.i0, self.r0 =U0[0], U0[1], U0[2]
        self.i_train =self.i[:indx]
        self.t_train =self.t[:indx]
        ##Testing
        self.i_test =self.i[indx:]
        self.t_test =self.t[indx:]
        self.i0t =self.i_test[0:1,:]
        self.r0t =np.array([[0.0]])
        self.s0t =self.N-self.i0t- self.r0t
        self.s0t =self.s0t.reshape((-1,1))
        self.weights, self.biases =self.w_b(self.n_layers)
        self.beta =tff.Variable([bi], dtype=tff.float32)
        self.gamma =tff.Variable([gi], dtype=tff.float32)
        ##Get placeholders as containers for input and output
        self.sess =tff.Session(config=tff.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.t_t =tff.placeholder(tff.float32, [None, 1])
        self.i_t =tff.placeholder(tff.float32, [None, 1])
        self.s0_t =tff.placeholder(tff.float32, [None, 1])
        self.i0_t =tff.placeholder(tff.float32, [None, 1])
        self.r0_t =tff.placeholder(tff.float32, [None, 1])
        ##Placeholders and data containers
        self.tf_t =tff.placeholder(tff.float32, [None, 1])
        self.s_pred, self.i_pred, self.r_pred=self.sir_network(self.t_t)
#         self.beta_pred = self.nn_beta(self.t_t)
        self.s0_pred =self.s_pred[0]
        self.i0_pred =self.i_pred[0]
        self.r0_pred =self.r_pred[0]
        self.e1,self.e2, self.e3=self.residual_sir(self.tf_t)
         # Loss: Initial Data
        self.lossUU0 = tff.reduce_mean(tff.square(self.s0_t - self.s0_pred)) + \
            tff.reduce_mean(tff.square(self.i0_t - self.i0_pred)) + \
            tff.reduce_mean(tff.square(self.r0_t - self.r0_pred)) 
        ##Data
        self.lossD = tff.reduce_mean(tff.square(self.i_t-self.i_pred))                   
        ##Residual
        self.lossR= tff.reduce_mean(tff.square(self.e1-0.0))+\
                    tff.reduce_mean(tff.square(self.e2-0.0))+\
                    tff.reduce_mean(tff.square(self.e3-0.0))
                    
               
        
        self.loss =self.lossUU0+self.lossD+self.lossR
        self.opt =tff.train.AdamOptimizer().minimize(self.loss)
    
        init =tff.global_variables_initializer()
        self.sess.run(init)                  
        ##Initialize weights and biases
    def xzavier(self,dim):
        d1,d2 =dim[0], dim[1]
        std =np.sqrt(2.0/(d1+d2))
        return tff.Variable(tff.truncated_normal([d1,d2], stddev=std), dtype=tff.float32)
    ##Apply to all wights
    def w_b(self, n_layers):
        l=n_layers
        weights =[self.xzavier([l[j], l[j+1]]) for j in range(0, len(l)-1)]
        biases =[tff.Variable(tff.zeros([1, l[j+1]], dtype =tff.float32),dtype=tff.float32) for j in range(0,len(l)-1)]
        return weights,biases
   
    #Define the neural network
    def network(self,t, weights, biases):
        M=len(weights)+1
        z=2.0*(t-self.lb)/(self.ub-self.lb)-1.0
        for i in range(0, M-2):
             z =tff.nn.tanh(tff.matmul(z, weights[i])+biases[i])
        y_pred =tff.nn.softplus(tff.matmul(z, weights[-1])+biases[-1])
        return y_pred
    
    def sir_network(self,t):
        out =self.network(t, self.weights, self.biases)
        s, i, r =out[:,0:1], out[:,1:2], out[:,2:3]
        return s, i, r

    
    def residual_sir(self,t):
        beta, gamma, v, eta =self.beta, self.gamma, self.v, self.eta
        s, i,r =self.sir_network(t)
        s_t =tff.gradients(s, t, unconnected_gradients='zero')[0]
        i_t =tff.gradients(i, t, unconnected_gradients='zero')[0]
        r_t =tff.gradients(r, t, unconnected_gradients='zero')[0]
        N=self.N
        e1 =s_t +(beta*s*i)/N +v*eta*s
        e2 =i_t -(beta*s*i)/N +gamma*i
        e3 =r_t -gamma*i-v*eta*s
        return e1, e2, e3
    def callbacks(self, loss, beta, gamma):
        print('Loss: {}, beta: {}, gamma: {}'.format(loss,beta,gamma))   
    def train(self, epochs):
        train_dic ={self.t_t: self.t_train, self.i_t:self.i_train,  self.tf_t:self.tf,
                   self.s0_t:self.s0, self.i0_t:self.i0, self.r0_t:self.r0}
        test_dic ={self.t_t: self.t_test, self.i_t:self.i_test,  self.tf_t:self.tf,
                   self.s0_t:self.s0t, self.i0_t:self.i0t, self.r0_t:self.r0t}
        start_time = time.time()
        for i in range(epochs+1):
            self.sess.run(self.opt, train_dic)
            self.sess.run(self.opt, test_dic)
            if i%100==0:
                elapsed = time.time() - start_time
                loss_v= self.sess.run(self.loss, train_dic)
                loss_v1= self.sess.run(self.loss, test_dic)
                beta_v=self.sess.run(self.beta)
                gamma_v=self.sess.run(self.gamma)
                print('Epoch: %d, Train Loss:%.3e, Test Loss:%.3e, beta: %.2f, gamma: %.2f, Time: %.2f'%(i, loss_v, loss_v1, beta_v, gamma_v,elapsed))
                start_time = time.time()
    def predict(self,t_hat):
        tr_dic={self.t_t: t_hat}
        s_hat =self.sess.run(self.s_pred, tr_dic)
        i_hat =self.sess.run(self.i_pred, tr_dic)
        r_hat =self.sess.run(self.r_pred, tr_dic)
        return s_hat, i_hat, r_hat 
