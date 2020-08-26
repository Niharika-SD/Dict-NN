# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:03:27 2020

@author: niharika-shimona
"""

#torch libs
import torch


import scipy
import numpy as np

from Helpers import err_compute,func,func_der


def update_basis(B,corr,C,model,D,lamb,Y,gamma,lambda_1,lambda_2,lambda_3):

   "update the basis term B using proximal gradient descent"

   #parameters   
   num_iter_max = 30 
   t = 0.0001

   #pre-allocate
   err_inner = np.zeros((num_iter_max,1))

   #main loop
   for iter in range(num_iter_max):
  
       #pre-allocate gradient variable
       DG = torch.zeros(B.size())
       
       for j in range(corr.size()[0]):
           
           #input variables
           Corr_j = corr[j,:,:]
           D_j = D[j,:,:]
           lamb_j = lamb[j,:,:]
           D_j_T = torch.transpose(D_j,0,1)
           
           #graident direction
           T1 = 2*(torch.mm((torch.mm(B,D_j_T)-Corr_j),D_j))
           T2 = torch.mm(D_j,torch.diagflat(C[:,j]))
           T3 = torch.mm(torch.mm(B,torch.diagflat(C[:,j])),torch.diagflat(C[:,j]))
           T4 = torch.mm(lamb_j,torch.diagflat(C[:,j]))
           DG = DG + T1 - T2 + T3 - T4 
       
       #proximal update       
       X_mat = B - t*DG/lambda_1
       Mult = (torch.max(torch.abs(X_mat)-t*torch.ones(X_mat.size()),torch.zeros(X_mat.size())))
       B = torch.sign(X_mat)*Mult

       #check exit condition
       if(iter == 0):
        
          DG_init = DG;

       err_inner[iter] = err_compute(corr,B,C,model,Y,D,lamb,gamma,lambda_1,lambda_2)

       print('At B iteration %d || Error: %1.3f' %(iter,err_compute(corr,B,C,model,Y,D,lamb,gamma,lambda_1,lambda_2)) )   
 
       if ((iter>1) and (((torch.norm(DG)/torch.norm(DG_init))< 10e-03) or (err_inner[iter]>err_inner[iter-1]))):
      
          if ( (err_inner[iter]>err_inner[iter-1])):

              print("Exiting due to increase in function value- adjust learning rate")
              
          break
      

   return B
   
        
def coefficient_update(C,model,D,B_upd,Y,lamb,gamma,lambda_2):
        
    "L-BFGS update for the coefficient matrix"
        
    #decoupled across patients    
    for n in range(C.size()[1]):
        
        #pre-allocate and initialize
        C_n_init = torch.reshape(C[:,n],(C.size()[0],1)).clone().numpy()
        C_n_init = np.asarray(C_n_init)
       
       #input variables
        lamb_n = lamb[n,:,:]
        D_n = D[n,:,:]
        y_n = Y[n,:]

        #non-neg constraints
        mybounds = [(1e-8,np.inf)]*np.shape(C_n_init)[0]

        #update for single patient
        C_n_upd = scipy.optimize.minimize(func, x0=C_n_init ,args=(D_n,lamb_n,y_n,B_upd,gamma,lambda_2,model), method='L-BFGS-B',
                                            bounds=mybounds,jac=func_der)
        
        #cast as torch tensors        
        if (n==0):
         
            m = np.shape(C_n_upd.x)[0]
            C_n_upd = np.reshape(np.asarray(C_n_upd.x),(m,1))
            C_upd = torch.from_numpy(C_n_upd)

        else:
            
            m = np.shape(C_n_upd.x)[0]
            C_n_upd = np.reshape(np.asarray(C_n_upd.x),(m,1))
            C_upd = torch.cat((C_upd,torch.from_numpy(C_n_upd)),1)
      
        print("Patient  %d Optimised " %(n))
        
    return C_upd.float()
        
        
   
def train_NN(corr,B_upd,C,Y,D,lamb,model,gamma,lambda_2,lambda_3,epochs,lr_nn):
     
    "Update neural network weights"
        
    model.train() #put in train mode
    
    C_T = torch.transpose(C,0,1)
    batch_size = 16 #batch size -tunable
    M = C_T.size()[0]
    
    # define optimizer and scheduler object
    optimizer = torch.optim.Adam(model.parameters(),lr =lr_nn, weight_decay = lambda_3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=1, last_epoch=-1)
    
    for epoch in range(epochs):
            
      permutation = torch.randperm(C_T.size()[0]) #shuffle data

      for i in range(0,C_T.size()[0]-1, batch_size):
          
          model.train() # model is updating batch norm parameters and implementing dropouts here 
          
          optimizer.zero_grad() #clear grads
          
          #batch selection
          if(i+batch_size >  (M-1)):
              
              inc = M

          elif(i ==  M - 1):
              
              break
          
          else:
              
              inc = i+batch_size
         
          
          #batch variables
          indices = permutation[i:inc]
          batch_x, batch_y = C_T[indices], Y[indices]
          y_pred = model.forward(batch_x.float())  #forward pass
          
          #Batchwize Backpropagation
          mask = batch_y>0 #mask targets for missing batch scores
          loss = gamma*(torch.norm(mask.mul(y_pred-batch_y))**2) #network loss
          loss.backward(retain_graph=True) #backprop
          
          optimizer.step() #update weight params
       
      #update learning rate
      scheduler.step()
     
      #print epoch loss
      mask_epoch = Y>0     #mask for missing targets   
      print('Epoch: %d, ||  Loss: %1.3f ' %(epoch, gamma*torch.norm(mask_epoch.mul(model.forward(C_T)-Y))**2))

    del optimizer
    del scheduler
    
    model.eval() #eval mode
    
    return model
        
        
def update_constraints(corr,lamb,B,C,lr):
    
    "Augmented Lagrangian updates for constraint variables"
    
    num_iter_max = 100 #max iter
    
    #pre-allocate
    lamb_upd = torch.zeros(lamb.size())
    D_upd = torch.zeros((corr.size()[0],B.size()[0],C.size()[0]))
    
    #loop over patients
    for k in range(lamb.size()[0]):
      
        #input variables
        Corr_k = corr[k,:,:]
        lamb_k = lamb[k,:,:]
               
        #loop over iter
        for c in range(num_iter_max):
               
            #closed form primal update   
            T1 = (torch.mm(B,torch.diagflat(C[:,k]))+ 2*torch.mm(Corr_k,B) - lamb_k)
            T2 =  torch.inverse(torch.eye(B.size()[1])+2*(torch.mm(torch.transpose(B,0,1),B)))
            D_k = torch.mm(T1,T2)
       
            #gradient ascent -lagrangian dual update  
            lamb_k = lamb_k + (0.5**(c-1))*lr*(D_k - torch.mm(B,torch.diagflat(C[:,k])))
        
           #check exit condition     
            if (c ==0):
                
                grad_norm_init = torch.norm(D_k - B.mm(torch.diagflat(C[:,k])))
                
            conv = torch.norm(D_k - torch.mm(B,torch.diagflat(C[:,k])))
            
            if (conv/grad_norm_init<10e-03):
               
               break;
      
        #patient wise update
        lamb_upd[k,:,:]= lamb_k
        D_upd[k,:,:] = D_k
    
    return D_upd,lamb_upd