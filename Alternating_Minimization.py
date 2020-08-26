# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:25:43 2020

@author: niharika-shimona
"""

import torch
from torch.autograd import Variable


from copy import copy
import numpy as np
import time


#my libs
from Optimization_Modules import update_basis, coefficient_update, train_NN, update_constraints
from Helpers import err_compute



def alt_min_main(corr,B_init,C_init,model_init,D_init,lamb_init,Y,gamma,lambda_1,lambda_2,lambda_3):
    
    "Main iteration module for alternating minization"
    
    #initialize unknowns
    B_old = B_init #basis
    C_old = C_init #temporal coeffs
    model_old = model_init #lstm-ann
    D_old = D_init #temporal constraints
    lamb_old = lamb_init #lagrangians
    
    #params
    num_iter_max = 50
    thresh = 1e-04 #exit thresh
    
    #pre-allocate
    err_out = np.zeros((num_iter_max,1))
    
    for iter in range(num_iter_max):

        # init err
        err_out[iter] = err_compute(corr,B_old,C_old,model_old,Y,D_old,lamb_old,gamma,lambda_1,lambda_2)        
        print(" At iteration: %d || Error: %1.3f  "  %(iter, err_out[iter]) )    
            
        #adjusts initial LR
            
        if (iter==0):
                
               epochs = 100
               lr_nn = 0.001
               
        elif (iter>0 and iter<10):
             
               epochs = 50
               lr_nn = 0.0005
               
        else:
            
             epochs = 50
             lr_nn = 0.00005
              
                        
        # run one alternating min step      
        [B,C,model,D,lamb] = alt_min_step(corr,B_old,C_old,model_old,D_old,lamb_old,Y,gamma,lambda_1,lambda_2,lambda_3,epochs,lr_nn) 
        
        
        # variable updates
        B_old = B
        C_old = C
        D_old = D
        lamb_old = lamb
        model_old = copy(model)
        
        # check exit conditions
        if((iter>5) and( (abs((err_out[iter]-err_out[iter-1])) < thresh)  or (err_out[iter]-err_out[iter-5]>30))):
           
            if(err_out[iter]>err_out[iter-1]): #fail safe              
                print(' Exiting due to increase in function value, at iter ' ,iter, ' Fix convergence- try adjusting learning rates and schedules')       
                
            break

    return B,C,model,D,lamb,err_out,iter
    
    

def alt_min_step(corr,B,C,model,D,lamb,Y,gamma,lambda_1,lambda_2,lambda_3,epochs,lr_nn):
    
   "Given the current values of the iterates, performs a single step of alternating minimization"

   ########
   "Basis update"
   
   print('Optimise B ')   
  
   B_upd = update_basis(B,corr,C,model,D,lamb,Y,gamma,lambda_1,lambda_2,lambda_3) 

   print(" At final B iteration || Error: %1.3f " %(err_compute(corr,B_upd,C,model,Y,D,lamb,gamma,lambda_1,lambda_2))) 


   ########   
   "Coefficients Update"

   print('Optimise C ')
  
   t0 = time.time()
   C_upd = coefficient_update(C,model,D,B_upd,Y,lamb,gamma,lambda_2) 
   print('{} seconds'.format(time.time() - t0)) 
   print(" Step C || Error: %1.3f"  %(err_compute(corr,B,C_upd,model,Y,D,lamb,gamma,lambda_1,lambda_2)))   


   ########
   "ANN weight update"
   print('Optimise Theta')

   model_upd = train_NN(corr,B_upd,C_upd,Y,D,lamb,model,gamma,lambda_2,lambda_3,epochs,lr_nn)
   
   print(" Step Theta || Error: %1.3f " %(err_compute(corr,B_upd,C_upd,model_upd,Y,D,lamb,gamma,lambda_1,lambda_2)))
  
    ########
   "Constraint variable updates "
   t0 = time.time()
   
   lr = 0.001
   [D_upd,lamb_upd] = update_constraints(corr,lamb,B,C,lr)
   print('{} seconds'.format(time.time() - t0))
   
   print(" Step D,lamb || Error: %1.3f " %(err_compute(corr,B_upd,C_upd,model_upd,Y,D_upd,lamb_upd,gamma,lambda_1,lambda_2)))
   return B_upd,C_upd,model_upd,D_upd,lamb_upd