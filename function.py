import pandas as pd
import numpy as np
import scipy as sp
from sklearn import metrics

import matplotlib.pyplot as plt

import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


import pandas as pd
import numpy as np
import scipy as sp

class ALS():
    def __init__(self, RDmatrix ,trainOmega , testOmega ):
        self.RDmatrix = RDmatrix
        self.trainOmega = trainOmega
        self.testOmega = testOmega
        self.drug = RDmatrix.shape[0]
        self.disease = RDmatrix.shape[1]
        self.X = None
        self.Y = None
        self.P = None
        self.C = None
        self.predict = None

        
    def ALScalculate( self , latent_dim = 10 , alpha = 40 , r_lambda = 1,residual = None ):
        self.X = np.random.rand(self.drug, latent_dim ) * 0.5
        self.Y = np.random.rand(self.disease,latent_dim) * 0.5
        self.P = np.copy(self.trainOmega.mul(self.RDmatrix))
        self.P[self.P>0] = 1
        
        self.predict_errors = []
        self.confidence_errors = []
        self.regularization_list = []
        self.total_losses = []
        
        
        if(residual is not None):
            self.C = np.asarray(alpha*self.trainOmega + residual)
        else:
            self.C = np.asarray(alpha*self.trainOmega)
            
        
        
        for i in range(5):
            if i!=0:   
                self.optimize_user(self.X, self.Y, self.C, self.P, self.drug, latent_dim, r_lambda)
                self.optimize_item(self.X, self.Y, self.C, self.P, self.disease, latent_dim, r_lambda)
            predict = np.matmul(self.X, np.transpose(self.Y))
            predict_error, confidence_error, regularization, total_loss = self.loss_function(self.C, self.P, predict, self.X, self.Y, r_lambda)

            self.predict_errors.append(predict_error)
            self.confidence_errors.append(confidence_error)
            self.regularization_list.append(regularization)
            self.total_losses.append(total_loss)

            print('----------------step %d----------------' % i)
            print("predict error: %f" % predict_error)
            #print("confidence error: %f" % confidence_error)
            print("regularization: %f" % regularization)
            print("total loss: %f" % total_loss)
        
        self.predict = np.matmul(self.X, np.transpose(self.Y))
        
        
    def AUC(self):
        if(self.RDmatrix is not None):
            label = self.RDmatrix.values.flatten()[(self.testOmega>0).values.flatten()]
            pred = self.predict.flatten()[(self.testOmega>0).values.flatten()]
        if(self.predict is None):
            print("please do ALScalculate")
            return
        fpr , tpr , thresholds = metrics.roc_curve(label,pred)
        print(metrics.auc(fpr,tpr))
        plt.plot(fpr,tpr)


    def loss_function(self, C, P, xTy, X, Y, r_lambda):
        predict_error = np.square(P - xTy)
        confidence_error = np.sum(C * predict_error)
        regularization = r_lambda * (np.sum(np.square(X)) + np.sum(np.square(Y)))
        total_loss = confidence_error + regularization

        return np.sum(predict_error), confidence_error, regularization, total_loss

    def optimize_user(self, X, Y, C, P, nu, nf, r_lambda):
        yT = np.transpose(Y)
        for u in range(nu):
            Cu = np.diag(C[u])
            yT_Cu_y = np.matmul(yT, Y)
            lI = np.dot(r_lambda, np.identity(nf))
            yT_Cu_pu = np.matmul(yT, P[u])
            X[u] = np.linalg.solve(yT_Cu_y+ lI, yT_Cu_pu)

    def optimize_item(self, X, Y, C, P, ni, nf, r_lambda):
        xT = np.transpose(X)
        for i in range(ni):
            Ci = np.diag(C[:, i])
            xT_Ci_x = np.matmul(xT, X)
            lI = np.dot(r_lambda, np.identity(nf))
            xT_Ci_pi = np.matmul(xT, P[:, i])
            Y[i] = np.linalg.solve(xT_Ci_x + lI, xT_Ci_pi)



class Make_Latent_Matrix(Module):
    

    def __init__(self, num_row , num_col , device ):
        super(Make_Latent_Matrix, self).__init__()
        self.in_features = num_row
        self.out_features = num_col
        self.weight = Parameter(torch.FloatTensor(np.random.rand( num_row , num_col)/3.).to(device))
        

    def forward(self):
        
        return self.weight            
            

        
        
        
class RALA_back():
    def __init__(self, RDmatrix ,trainOmega , testOmega , GPU_NUM = 0 ,latent_dim = 10 ):
    
        self.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        self.A = Make_Latent_Matrix(RDmatrix.shape[0],latent_dim , self.device)
        self.B = Make_Latent_Matrix(latent_dim,RDmatrix.shape[1] , self.device)
        self.RDmatrix = RDmatrix
        self.trainOmega = trainOmega
        self.testOmega = testOmega
        self.checkCost = None
        
        self.label = torch.FloatTensor(trainOmega.values).mul( torch.FloatTensor(RDmatrix.values) ).to(self.device)
        self.trainOmega_torch = torch.FloatTensor(trainOmega.values).to(self.device)
        self.predict = None
        
    def RALA_calculate_back(self , Alter_epoch = 5 , backprop_epoch = 1000 , lambda_ = 0 ,learning_rate = 0.001):
        A_optimizer = torch.optim.Adam(self.A.parameters(), lr=learning_rate)
        B_optimizer = torch.optim.Adam(self.B.parameters(), lr=learning_rate)
        
        for epoch in range(Alter_epoch):
            
            self.A.requires_grad_(False)
            self.B.requires_grad_(True)
            
            for i in tqdm(range(backprop_epoch)):

                cost = F.l1_loss( self.trainOmega_torch.mul( torch.mm(self.A(),self.B()) ) , self.label ) + lambda_*( torch.norm(self.B(),1))
                
                B_optimizer.zero_grad()
                cost.backward()
                B_optimizer.step()
                
            self.A.requires_grad_(True)
            self.B.requires_grad_(False)
            
        
            for j in tqdm(range(backprop_epoch)):
                cost = F.l1_loss( self.trainOmega_torch.mul( torch.mm(self.A(),self.B()) ) , self.label ) + lambda_*( torch.norm(self.A(),1)) 
                
                A_optimizer.zero_grad()
                cost.backward()
                A_optimizer.step()    
                cost
            
            print('Epoch {:4d}/{} cost : {:6f}'.format(epoch+1, Alter_epoch,cost.item()))
                
        self.predict = torch.mm(self.A(),self.B()).to(device = 'cpu').detach().numpy()
            


            
    def AUC(self):
        if(self.RDmatrix is not None):
            label = self.RDmatrix.values.flatten()[(self.testOmega>0).values.flatten()]
            pred = self.predict.flatten()[(self.testOmega>0).values.flatten()]
        if(self.predict is None):
            print("please do RALA_calculate_back")
            return
        fpr , tpr , thresholds = metrics.roc_curve(label,pred)
        print(metrics.auc(fpr,tpr))
        plt.plot(fpr,tpr)
        
        
class RALA_LP():
    def __init__( self, RDmatrix ,trainOmega , testOmega ):
        self.RDmatrix = RDmatrix
        self.trainOmega = trainOmega
        self.testOmega = testOmega
        self.drug = RDmatrix.shape[0]
        self.disease = RDmatrix.shape[1]
        self.predict = None
        self.A = None
        self.B = None
        self.RDmatrixT = None
        self.trainOmegaT = None
        self.AT = None
        self.BT = None
        
    def RALA_calculate_LP(self , Alter_epoch = 5,latent_dim = 10 , r_lambda = 1):
        self.A = np.random.rand(self.drug , latent_dim)/3
        self.B = np.random.rand(latent_dim , self.disease)/3
        
        self.RDmatrixT = self.RDmatrix.T
        self.trainOmegaT = self.trainOmega.T
        self.AT = self.A.T
        
        for j in range(Alter_epoch):
            print("epoch: ",j+1)
            for i in tqdm(range(self.RDmatrix.shape[1])):
                self.B[:,i] = self.l1_fit(self.A, self.RDmatrix.iloc[:,i], self.trainOmega.iloc[:,i],1)


            self.BT = self.B.T

            for i in tqdm(range(self.RDmatrixT.shape[1])):    
                self.AT[:,i] = self.l1_fit(self.BT, self.RDmatrixT.iloc[:,i], self.trainOmegaT.iloc[:,i],1)


            self.A = self.AT.T
        
        self.predict = self.A@self.B
            
            
            
    def l1_fit(self , U, v, val ,lamb_da = 1):
        """
        Find a least absolute error solution (m, k) to U * m + k = v + e.
        Minimize sum of absolute values of vector e (the residuals).
        Returned result is a dictionary with fit parameters result["m"] and result["k"]
        and other information.
        """
        U = np.array(U)
        v = np.array(v)                                      # 람다도 설정해야지
        # n is the number of samples
        n = len(v)
        s = U.shape
        assert len(s) == 2
        assert s[0] == n
        # d is the number of dimensions
        d = s[1]
        regular = np.ones(d) * lamb_da
        I = np.diag( np.concatenate([np.ones(n) , regular ]) )
        I = np.identity(n+d)   # np.identity(len(v))      n개의 t랑 d개의 m (정규식 절대값 포함)
        Im = np.identity(d)
        U_ = np.vstack([U,
                     Im])

        A = np.vstack([
                np.hstack([-I, U_]),
                np.hstack([-I, -U_])
            ])
        val = np.asarray(val)
        c = np.hstack([ val ,np.ones(d), np.zeros(d)])       # min sum( t ) + sum( m ) + zeros는 min에 상관무라는 뜻
        b = np.hstack([v, np.zeros(d) ,-v ,np.zeros(d)])
        bounds = [(0, None)] * I.shape[0] + [(None, None)] * (d)
        options = {"maxiter": 50 , 'tol':1e-1}
        # Call the linprog subroutine.
        r = scipy.optimize.linprog(c, A, b,  options=options)
        # Extract the interpolation result from the linear program solution.
        x = r.x
        m = x[I.shape[0]:I.shape[0]+d]
        v_predicted = np.dot(U, m) 
        residuals = v - v_predicted
        # For debugging store all parameters, intermediates and results in returned dict.
        
        return m
    
    def AUC(self):
        if(self.RDmatrix is not None):
            label = self.RDmatrix.values.flatten()[(self.testOmega>0).values.flatten()]
            pred = self.predict.flatten()[(self.testOmega>0).values.flatten()]
        if(self.predict is None):
            print("please do ALScalculate")
            return
        fpr , tpr , thresholds = metrics.roc_curve(label,pred)
        print(metrics.auc(fpr,tpr))
        plt.plot(fpr,tpr)