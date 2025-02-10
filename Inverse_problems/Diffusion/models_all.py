import numpy as np
import torch
from torch import nn, optim, autograd
from torch.nn import functional as F

class Ada_CL():
    def __init__(self,e_k,q_k,s_k,device):

        self.e_k = e_k
        self.q_k = q_k
        self.s_k = s_k
        self.device = device
        
    def default_log_alpha_k_function(self,q_k,s_k,rho_k):

        if rho_k >= 0.98:
            rho_k = s_k

        if rho_k <= 0.02:
            rho_k = s_k

        alpha_k = q_k*np.log((rho_k*(1-s_k))/(s_k*(1-rho_k)))
        
        return alpha_k    
        
    def Difficulty_Measurer(self,error_sequence,weight_sequence):
        
        data_index = np.arange(len(error_sequence))
        
        Easier_Point_index = data_index[np.where(error_sequence<=self.e_k)[0]].astype(int)
        Harder_Point_index = data_index[np.where(error_sequence>self.e_k)[0]].astype(int)

        beta_sequence = np.zeros_like(error_sequence)
        beta_sequence[Easier_Point_index] = 1
        
        #print(Easier_Point_index)
        Easier_Point_weight = weight_sequence[Easier_Point_index]
        Harder_Point_weight = weight_sequence[Harder_Point_index]
        
        rho_k = np.sum(Harder_Point_weight)
        
        return rho_k,beta_sequence
    
    def Training_Scheduler(self,error_sequence,weight_sequence,alpha_k_function = 'default',record = 'not'):
        
        rho_k,beta_sequence = self.Difficulty_Measurer(error_sequence,weight_sequence)
        
        if alpha_k_function == 'default':
            alpha_k_function = self.default_log_alpha_k_function

        alpha_k = alpha_k_function(self.q_k,self.s_k,rho_k)
        
        weight_sequence_new = weight_sequence*np.exp(-alpha_k*beta_sequence)
        weight_sequence_new = weight_sequence_new/np.sum(weight_sequence_new)
    
        if record == 'yes': 
            return weight_sequence_new,rho_k,beta_sequence
        else:
            return weight_sequence_new

    def Training_Scheduler_torch(self,error_sequence,weight_sequence,alpha_k_function = 'default',record = 'not'):
        
        if self.device == "cuda":
            error_sequence_numpy = error_sequence.cpu().detach().numpy()  
        else:
            error_sequence_numpy = error_sequence.detach().numpy()  
            
        rho_k,beta_sequence = self.Difficulty_Measurer(error_sequence_numpy,weight_sequence)
        
        if alpha_k_function == 'default':
            alpha_k_function = self.default_log_alpha_k_function

        alpha_k = alpha_k_function(self.q_k,self.s_k,rho_k)
        
        weight_sequence_new = weight_sequence*np.exp(-alpha_k*beta_sequence)
        weight_sequence_new = weight_sequence_new/np.sum(weight_sequence_new)
        
        if self.device == "cuda":
            weight_sequence_torch = torch.from_numpy(weight_sequence_new).cuda()
            error_new = torch.sum(weight_sequence_torch * error_sequence)
        else:
            weight_sequence_torch = torch.from_numpy(weight_sequence_new)
            error_new = torch.sum(weight_sequence_torch * error_sequence)  
            
        if record == 'yes': 
            return error_new,weight_sequence_new,rho_k,beta_sequence
        else:
            return error_new,weight_sequence_new
        
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class hidden_layers(nn.Module):
    def __init__(self,input_number,output_number):
        super(hidden_layers, self).__init__()
        self.layer = nn.Linear(input_number,output_number)
    def forward(self, x):
        x = self.layer(x)
        x = torch.tanh(x)
        return x

class NN_H2 (nn.Module):
    def __init__(self,in_N, width, depth, out_N):

        super(NN_H2, self).__init__()
        self.in_N = in_N
        self.width = width
        self.depth = depth
        self.out_N = out_N

        self.stack = nn.ModuleList()

        self.stack.append(hidden_layers(in_N, width))

        for i in range(depth):
            self.stack.append(hidden_layers(width, width))

        self.stack.append(nn.Linear(width, out_N))
        
        
    def forward(self, x):
        for m in self.stack:
            x = m(x)
        return x

class get_discriminator(nn.Module):
    def __init__(self,in_N, width, depth, out_N):
        #depth = layers-2
        super(get_discriminator, self).__init__()
        self.in_N = in_N
        self.width = width
        self.depth = depth
        self.out_N = out_N

        self.stack = nn.ModuleList()

        self.stack.append(hidden_layers(in_N, width))

        for i in range(depth):
            self.stack.append(hidden_layers(width, width))

        self.stack.append(nn.Linear(width, out_N))
        
        
    def forward(self, x):
        for m in self.stack:
            x = m(x)
            x = torch.sigmoid(x)
        return x 

def relative_l2(u_pred,u_real):
    l2 = np.linalg.norm(u_real-u_pred,2)/np.linalg.norm(u_real,2)
    return l2


