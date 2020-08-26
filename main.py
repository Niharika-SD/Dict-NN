# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:00:34 2020

@author: niharika-shimona
"""

import sys
import numpy as np
import os

# torch
import torch
import pickle

#scipy
import scipy.io as sio


#Matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#my libs
from Alternating_Minimization import alt_min_main
from ANN import ANN
from Helpers import init_const
from Quadratic_Solver import Quad_Solve_Coefficients


if __name__ == '__main__':      
    
    #params Dict-NN
    gamma = 1 #regression penalty
    lambda_1 = 10 # sparsity penalty 
    lambda_2 = 0.7 # coefficient penalty 
    lambda_3 = 0.00001 # weight decay
    net = 8 #number of networks 
    
    #params LSTM-ANN
    hidden_size = 40 #hidden layer size
    num_targets = 3 #no of targets
    input_size = net

    
    path_name = '/home/niharika-shimona/Documents/Projects/Autism_Network/Dict-NN/Data/'
    output_dirname = path_name + '/Outputs/'

    if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)
    
    #initlialize
    test_pred_Y = []
    test_Y = []
    train_pred_Y = []
    train_Y = []
    
    #print to logfile
    log_filename = output_dirname + 'logfile.txt'
    log = open(log_filename, 'w')
    sys.stdout = log   
      
    #load data
    data = sio.loadmat(path_name +'/data.mat')

    #train
    corr_train = torch.from_numpy(data['corr_train']).float()
    Y_train = torch.from_numpy(np.asarray(data['Y_train'],dtype=np.float32)).float()
    
    #test
    corr_test = torch.from_numpy(data['corr_test']).float()     
    Y_test = torch.from_numpy(np.asarray(data['Y_test'],dtype=np.float32)).float()
    
    
    ##NOTE: Unknown scores have been set to zero and are not used in estimation of parameters or backpropagation
    
    #initialization
    corr_mean = torch.mean(corr_train,0)       
    [D,V] = torch.eig(corr_mean,eigenvectors=True) #EVD of mean corr    
        
    B_init = V[:,0:net] #init B in eigenspace
    C_init = Quad_Solve_Coefficients(corr_train,B_init,lambda_2) #solve coeffs w/o regression
 
    [D_init,lamb_init] = init_const(B_init,C_init) #const init   
     
    model_init = ANN(input_size,hidden_size,num_targets) #model init

    #run optimization    
    [B_gd,C_gd,model_gd,D_gd,lamb_gd,err_out,iter] = alt_min_main(corr_train,B_init,C_init,model_init,D_init,lamb_init,Y_train,gamma,lambda_1,lambda_2,lambda_3)
        
    #performance evaluation
    
    model_gd.eval() #eval mode
    
    #test
    C_test  =  Quad_Solve_Coefficients(corr_test,B_gd,lambda_2)    
    test_pred_Y = model_gd.forward(C_test.transpose(0,1)).detach().numpy()
    test_Y = Y_test.detach().numpy()
    
    #train
    train_pred_Y = model_gd.forward(C_gd.transpose(0,1)).detach().numpy()
    train_Y = Y_train.detach().numpy()
   
    #assign unknown scores to zero
    train_pred_Y[train_Y==0] = 0
    test_pred_Y[test_Y==0] = 0
     
    fig,ax = plt.subplots()
    ax.plot(list(range(iter)),err_out[0:iter],'r')
       
    plt.title('Loss',fontsize=16)
    plt.ylabel('Error' ,fontsize=12)
    plt.xlabel('num of iterations',fontsize=12)
    plt.show()
    figname = output_dirname + 'Loss.png'
    fig.savefig(figname)   # save the figure to fil
    plt.close(fig)
       
    fig1,ax1 = plt.subplots()
    ax1 = plt.imshow(B_gd.detach().numpy(), cmap=plt.cm.jet,aspect='auto')
    plt.title('Recovered Networks')
    plt.show()
    figname1 = output_dirname +'B_gd.png'
    fig1.savefig(figname1,dpi=200)   # save the figure to fil
    plt.close(fig1)
    
    dict_save = {'model': model_gd, 'B_gd': B_gd, 'C_gd': C_gd,'C_test':C_test}
    filename_models =  output_dirname + 'Dict-NN.p'
    pickle.dump(dict_save, open(filename_models, "wb"))
       
    dict_cvf_per = {'Y_test': test_Y, 'Y_pred_train': train_pred_Y, 'Y_train': train_Y, 'Y_pred_test': test_pred_Y}
    sio.savemat(output_dirname+'Performance.mat',dict_cvf_per)
    
    