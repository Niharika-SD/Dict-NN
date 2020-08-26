# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:33:42 2020

@author: niharika-shimona
"""
#torch libs
import torch
from torch import optim

import numpy as np

#scipy libs
import scipy
from scipy import optimize

def func_test(params, *args):
    "Define a quadratic program at test time for coeffs"     
     
   #quad prog args            
    H_n = args[0].numpy()
    f_n = args[1].numpy()
    
    #params
    C_n = params
    C_n_T = np.transpose(C_n)

    #compute objective
    f_n_T = np.transpose(f_n)
    error_C_n = 0.5*C_n_T.dot(H_n.dot(C_n)) + f_n_T.dot(C_n)

    return error_C_n
    
def func_test_der(params, *args):
    
    "Define a quadratic program gradient at test time for coeffs"
    
    #args            
    H_n = args[0].numpy()
    f_n = args[1].numpy()
    
    #params    
    C_n = params

    grad_C_n = np.matmul(H_n,C_n) + f_n #compute gradient

    return grad_C_n

def get_input_optimizer(input_C):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS(input_C,lr=0.9)
    return optimizer
    
    
def Quad_Solve_Coefficients(corr_test,B_gd,lambda_2):

	"Quadratic Solver for coefficients at test time"
     
	for v in range(corr_test.size()[0]):
		
          # input variables
          Corr_v = corr_test[v,:,:]
		
          # quad term
          H_n = 2.0*(B_gd.transpose(0,1).mm(B_gd)).mm(B_gd.transpose(0,1).mm(B_gd)) + 2.0*lambda_2*torch.eye(B_gd.size()[1])
		
          #linear term
          X = torch.mm((torch.transpose(B_gd,0,1)),torch.mm(Corr_v,B_gd))
          f_n = -2.0*torch.diag(X)
          
          #init solution
          C_n_init = torch.zeros((B_gd.size()[1],1)).numpy()
		
          #non-negativity constraints
          mybounds = [(1e-8,None)]*np.shape(C_n_init)[0]

          #quad prog
          C_n_upd_test = scipy.optimize.minimize(func_test,x0=C_n_init,args=(H_n,f_n),method='L-BFGS-B', bounds=mybounds, jac=func_test_der)
		
          #cast as tensor
          if(v==0):

                m = np.shape(C_n_upd_test.x)[0]
                C_n_upd_test = np.reshape(np.asarray(C_n_upd_test.x),(m,1))
                C_upd_test = torch.from_numpy(C_n_upd_test)

          else:
              
              m = np.shape(C_n_upd_test.x)[0]
              C_n_upd_test = np.reshape(np.asarray(C_n_upd_test.x),(m,1))
              C_upd_test = torch.cat((C_upd_test,torch.from_numpy(C_n_upd_test)),1)
			
	
	return C_upd_test.float()