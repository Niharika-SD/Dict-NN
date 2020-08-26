# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:07:46 2020

@author: niharika-shimona
"""



import torch

import numpy as np
from numpy import linalg as LA


############################################################################### 
"Helper functions for main optimization modules"   


def init_const(B_init,C_init):
    
    "Inititalize constraint variables at start"

    #pre-allocate list variables
    D_init = torch.zeros(C_init.size()[1],B_init.size()[0],B_init.size()[1])
        
    for n in range(C_init.size()[1]):        
        
            D_init[n,:,:] = B_init.mm(torch.diagflat(C_init[:,n]))
            
            
    lamb_init = torch.zeros(D_init.size()).float() 
        
    return D_init,lamb_init
    
    
def err_compute(corr,B,C,model,Y,D,lamb,gamma,lambda_1,lambda_2):
    
    "Computes the error at the current main interation"

    #initialize
    fit_err = 0
    const_err = 0 
    
    for n in range(Y.size()[0]):
    
        #input variables
        Corr_n = corr[n,:,:]
        lamb_n = lamb[n,:,:]
        D_n = torch.mm(B,torch.diagflat(C[:,n]))
        
        B_T = torch.transpose(B,0,1)
        
        #update fit error
        X = Corr_n - torch.mm(D_n,B_T) 
        fit_err = fit_err + torch.norm(X)**2

        #update constraint error
        const = D[n,:,:] - D_n
        lamb_n_T = torch.transpose(lamb_n,0,1)
        const_err = torch.trace(torch.mm(lamb_n_T,const)) + torch.norm(const)**2 

    C_T =torch.transpose(C,0,1)
    est_Y = model.forward(C_T.float())
    mask = Y>0 #mask targets with missing data
    
    #compute total error  
    err = fit_err + (gamma*torch.norm(mask.mul(est_Y-Y))**2) + const_err 
        
    return err.detach().numpy()

def func(params, *args):

   "function definition for coeffs function patient wise"
    
   #arguments
   D_n = args[0].numpy() 
   lamb_n = args[1].numpy()
   y_n = args[2].numpy()
   B = args[3].numpy()
   gamma = args[4]
   lambda_2 = args[5]
   model = args[6]

   #param
   C_n = params
   
   m = np.shape(C_n)[0]

   #pre-allocate   
   C_n = np.reshape(C_n,(m,1))
   C_n_T = torch.transpose(torch.from_numpy(C_n),0,1)    
   y_pred = torch.zeros(np.size(y_n))
    
   #forward pass through network
   y_pred = model.forward(C_n_T.float()).detach().numpy()  
   mask = y_n>0 #mask for missing targets
    
   #network loss
   loss_network = LA.norm(np.multiply(y_pred-y_n,mask),'fro')**2
        
   #constraint loss     
   cons = D_n - np.matmul(B,np.diagflat(C_n))         
   constr_err = 0.5*LA.norm(D_n-np.matmul(B,(np.diagflat(C_n))),'fro')**2 + np.trace(np.matmul(lamb_n.T,cons))
 
   #total error    
   error = gamma*loss_network + lambda_2 * LA.norm(C_n,'fro')**2 + constr_err
   
   return error
   
def func_der(params, *args):
   
    "gradient definition for coeffs function patient wise"
    
    #arguments
    D_n = args[0].numpy()
    lamb_n = args[1].numpy()
    y_n = args[2]
    B = args[3].numpy()
    gamma = args[4]
    lambda_2 = args[5]
    model_upd = args[6]

    #param
    C_n = params
   
    m = np.shape(C_n)[0]
    
    #pre-allocate
    C_n = np.reshape(C_n,(m,1))
    C_n_T = torch.transpose(torch.from_numpy(C_n),0,1)
    C_n_T = torch.autograd.Variable(C_n_T,requires_grad=True)
    grad_C_nn = torch.zeros(C_n_T.size())
    
    # gradient from fitting and regularizer
    der_C = 2*lambda_2*C_n.T + np.multiply(C_n.T,np.diag(np.matmul(B.T,B))) - np.diag(np.matmul((D_n+lamb_n).T,B)).T
          
    #forward pass      
    y_pred = model_upd.forward(C_n_T.float())
    mask = y_n>0 #mask targets
    
    #backprop to compute ANN grads
    loss_y = torch.norm(mask.mul(y_pred-y_n))**2 
    loss_y.backward(retain_graph=True)
    grad_C_nn = grad_C_nn + C_n_T.grad.float() #update final grad
 
    #toal grad
    der_C = der_C.T + gamma*(grad_C_nn.detach().numpy()).T
 
    return np.asarray(der_C)
    
