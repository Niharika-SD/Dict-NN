# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:11:31 2020

@author: niharika-shimona
"""



# torch
import torch
from torch import  nn


class ANN(nn.Module):

    def __init__(self, input_size, hidden_size,num_targets):
        super(ANN, self).__init__()
        
        #params        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_targets = num_targets

        #FC definition
        self.dense1 = torch.nn.Linear(self.input_size,self.hidden_size, bias=True) #FC1
        self.dense2 = torch.nn.Linear(self.hidden_size,self.hidden_size, bias=True) #FC2
        self.dense3 = torch.nn.Linear(self.hidden_size,self.num_targets, bias=True) #FC3
        
        self.nl = torch.nn.ReLU()
 
    def forward(self, x):
       
       #forward pass definiton
       
        out = self.dense1(x) 
        out = self.nl(out)
        out = self.dense2(out)
        out = self.dense3(out)   
        
        return out
        
        
def init_weights(m):
    
    #init weights for ANN
    
    if type(m) == nn.Linear:
            torch.nn.init.xavier_normal(m.weight, gain=nn.init.calculate_gain('tanh'))
            m.bias.data.fill_(1e-02)
        
        
