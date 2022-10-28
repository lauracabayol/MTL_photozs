#!/usr/bin/env python

import torch
from torch import nn

from torch import nn
class Network_mtl(torch.nn.Module):
    def __init__(self,ncolors):
        super().__init__()
        
        self.block1= torch.nn.Sequential(nn.Linear(ncolors,100),nn.Dropout(0.02),nn.ReLU(),
                                 nn.Linear(100 ,150),nn.Dropout(0.02),nn.ReLU(),
                                 nn.Linear(150 ,200),nn.Dropout(0.02),nn.ReLU(),
                                 nn.Linear(200 ,150),nn.Dropout(0.02),nn.ReLU(),
                                 nn.Linear(150 ,100),nn.Dropout(0.02),nn.ReLU(),
                                 nn.Linear(100 ,50),nn.Dropout(0.02),nn.ReLU())
        
        self.zs = torch.nn.Sequential(nn.Linear(50 ,20),nn.ReLU(),nn.Linear(20 ,3))
        self.zerrs = torch.nn.Sequential(nn.Linear(50 ,20),nn.ReLU(),nn.Linear(20 ,3))
        self.alphass = torch.nn.Sequential(nn.Linear(50 ,20),nn.ReLU(),nn.Linear(20 ,3))

        self.fluxes = torch.nn.Sequential(nn.Linear(50 ,20),nn.ReLU(),nn.Linear(20 ,39))
                         
    def forward(self, img):
        hidden_space = self.block1(img)
    
        logalphas = self.alphass(hidden_space) 
        z = torch.abs(self.zs(hidden_space) )
        logzerr = self.zerrs(hidden_space)
        
        logalphas = logalphas - torch.logsumexp(logalphas,1)[:,None] 

        f = self.fluxes(hidden_space)
        
        return f, logalphas ,z,logzerr
    

                                      


