import numpy as np
import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
import torch.nn.init as init

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
        
        
class NetSetting:

    def __init__(self, input_dims, hidden_neurons_list, output_dims, hidden_activation, output_activation=None, initializer_method='xavier'):

        self.input_dims = input_dims
        self.hidden_neurons_list = hidden_neurons_list
        self.output_dims = output_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.initializer_method = initializer_method
        
def sigmoid_tanh(x_input):
    return torch.sigmoid(torch.tanh(x_input))

def get_activation_function(activation_name):

    if activation_name == 'tanh':
        return torch.tanh
    elif activation_name == 'sin':
        return torch.sin
    elif activation_name == 'relu':
        return torch.relu
    elif activation_name == 'sigmoid':
        return torch.sigmoid
    elif activation_name == 'sigmoid(tanh)':
        return sigmoid_tanh
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

class HiddenLayers(nn.Module):

    def __init__(self, net_settings, input_number, output_number):
        super(HiddenLayers, self).__init__()
        self.layer = nn.Linear(input_number, output_number)
        self.activation = get_activation_function(net_settings.hidden_activation)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

class get_mlp_pinn(nn.Module):

    def __init__(self, net_settings):
        super(get_mlp_pinn, self).__init__()
        self.net_settings = net_settings

        self.stack = nn.ModuleList()
        
        # First layer from input dimension to first hidden dimension
        self.stack.append(HiddenLayers(net_settings, net_settings.input_dims, net_settings.hidden_neurons_list[0]))
        
        # Additional hidden layers
        for i in range(1, len(net_settings.hidden_neurons_list)):
            self.stack.append(HiddenLayers(net_settings, net_settings.hidden_neurons_list[i-1], net_settings.hidden_neurons_list[i]))
        
        # Output layer
        self.stack.append(nn.Linear(net_settings.hidden_neurons_list[-1], net_settings.output_dims))

    def forward(self, x):
        for m in self.stack:
            x = m(x)
        if self.net_settings.output_activation:
            x = get_activation_function(self.net_settings.output_activation)(x)
        return x

def initialize_weights(model, method='xavier'):

    if method == 'xavier':
        for name, param in model.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
                
def compute_higher_order_derivatives(u, input_vars_list):

    grad = u
    for input_vars in input_vars_list:
        grad = autograd.grad(outputs=grad, inputs=input_vars,
                             grad_outputs=torch.ones_like(grad),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad
    

def relative_l2_torch(u_pred, u_real, p_value = 2):

    l2 = torch.norm(u_real - u_pred, p_value) / torch.norm(u_real, p_value)
    
    return l2.item()  

def relative_l2_numpy(u_pred, u_real, ord_value = 2):

    l2 = np.linalg.norm(u_real - u_pred, ord_value) / np.linalg.norm(u_real, ord_value)
    
    return l2  

def numpy_to_tensor(data, var_name, value_range_dim = None, to_torch = None, to_cuda = None, requires_grad = None):
    
    if value_range_dim is True:
        for col_i in range(data.shape[1]):
            min_val = np.min(data[:, col_i])
            max_val = np.max(data[:, col_i])
            print(f"{var_name}: Column {col_i}: range from {min_val} to {max_val}")
    
    if to_torch is True:
        data = torch.from_numpy(data).float()
        
    if to_cuda is True:
        data = data.cuda()
        
    if requires_grad is True:
        data.requires_grad_()

    return data