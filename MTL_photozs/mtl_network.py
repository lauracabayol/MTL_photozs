#!/usr/bin/env python
# encoding: UTF8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import os


from torch import nn, optim
from torch.optim import lr_scheduler
from .network import Network_mtl

class mtl_photoz:
    """Interface for photometry prection using neural networks."""
    
    # Here we estimate photometry on CPUs. This should be much
    # simpler to integrate and sufficiently fast.
    def __init__(self, zs=True, zs_NB=False, zs_zb=False, zs_NB_zb=False):
             
        self.zs = zs 
        self.zs_NB = zs_NB
        self.zs_zb = zs_zb
        self.zs_NB_zb = zs_NB_zb
        
        self.net = Network_mtl().cuda()
        
        if self.zs: self.w = 0
        if self.zs_NB: self.w = 1
        if self.zs_zb: self.w = 0
        if self.zs_NB_zb: self.w = 1
        
    def create_loader(self,dfbb,dfnb,cuts = 6):

        catalog_nb = dfnb[dfnb.index.isin(dfbb.paudm_id)]
        catalog_nb = 26-2.5*np.log10(catalog_nb)
        
        samps_BB_spec =  catalog_training[BB_list].values
        samps_NB_spec =  catalog_nb_train[NB_list].values

        samps_BB_colors_spec = samps_BB_spec[:,:-1] - samps_BB_spec[:,1:]
        samps_NB_colors_spec = samps_NB_spec[:,:-1] - samps_NB_spec[:,1:]
        
        if self.zs: zb_spec = catalog_training.target_zs.values
        if self.zs_NB: zb_spec = catalog_training.target_zs.values
        if self.zs_zb: zb_spec = catalog_training.target_zb.values
        if self.zs_NB_zb: zb_spec = catalog_training.target_zb.values
        
        field = zb_spec.copy()
        field = np.where(field== 0,0,1)
        
        samps_BB_colors_spec, samps_NB_colors_spec,zb_spec, field = \
        torch.Tensor(samps_BB_colors_spec), torch.Tensor(samps_NB_colors_spec), \
        torch.Tensor(zb_spec), torch.Tensor(field)
        data = TensorDataset(samps_BB_colors_spec, samps_NB_colors_spec, zb_spec,field)
        loader_train = DataLoader(data, batch_size=500, shuffle = True)      
        
        return loader_train
        
    def train_mtl(self,loader, epochs = 65):

        optimizer = optim.Adam(self.net.parameters(), lr=1e-3) #, weight_decay=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
        for epoch in range(epochs):

            for BB_flux, NB_flux_true, zb_true, field in loader_train:
                optimizer.zero_grad() 

                f, logalphas ,z,logzerr = net(BB_flux.cuda())
                zerr = torch.exp(logzerr)

                loss_z = logalphas - 0.5 * ((z - zb_true[:,None].cuda()) / zerr).pow(2) - logzerr
                loss_z  = torch.logsumexp(loss_z, 1)
                loss_z = field.cuda() * loss_z
                loss_z = -loss_z[loss_z!=0].mean()

                loss_pau = torch.abs(f - NB_flux_true.cuda())
                loss_pau = torch.nansum(loss_pau,1)
                loss_pau = loss_pau.mean()

                loss = self.w*loss_pau + loss_z

                loss.backward()
                optimizer.step()

            scheduler.step()

        return net
    
    def eval_mtl(self,test_colors):
        
        _,logalphas, z,logzerr = self.net(test_colors.cuda())

        alphas = torch.exp(logalphas)
        zb = (alphas * z).sum(1)
        zb,logzerr  = zb.detach().cpu().numpy(), logzerr.detach().cpu().numpy()

        return zb,logzerr

   