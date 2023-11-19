
import torch
from torch import nn
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# import itertools
import os
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

from dataclasses import dataclass, replace

from data import generate_points,TreeNode,gen_std_basis_DT,CustomDataset

from log_functions import log_features_dlgn,log_features_dlgn_sf,log_features_DLGN_kernel,feature_stats

from helper import plot_histogram,return_bound,return_data_elements

from models import DLGN,DLGN_SF,DLGN_Kernel,DNN

# from kernels import ScaledSig,gate_score,compute_npk,NPK

from copy import deepcopy

from train import train

import logging
logger = logging.getLogger(__name__)



@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hydra.utils.log.info("Logging from main")
    print("Running main function...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Access parameters via cfg
    data_config = cfg.data
    model_config = cfg.model
    train_config = cfg.train
    log_config = cfg.log

    data,train_dataloader,test_dataloader = return_data_elements(data_config, train_config)  

    if model_config.model_type == 'dlgn':
        model = DLGN(dim_in=data_config.dim_in, width=model_config.width, depth=model_config.depth, beta=model_config.beta, bias_fn=model_config.bias_fn, bias_vn=model_config.bias_vn)
    elif model_config.model_type == 'dlgn_sf':
        model = DLGN_SF(dim_in=data_config.dim_in, width=model_config.width, depth=model_config.depth, beta=model_config.beta, bias_fn=model_config.bias_fn, bias_vn=model_config.bias_vn)
    elif model_config.model_type == 'dlgn_kernel':
        model = DLGN_Kernel(dim_in=data_config.dim_in, width=model_config.width, depth=model_config.depth, beta=model_config.beta)
    elif model_config.model_type == 'dnn':
        model = DNN(dim_in=data_config.dim_in, dim_out=data_config.dim_out, width=model_config.width, depth=model_config.depth)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_config.lr)

    Train_losses,features_train,acc_dict = train(model,loss_fn,optimizer,train_dataloader,data,log_features_dlgn,log_config,train_config,print_std=True)

    torch.save(model.state_dict(), 'model.pt')
    logger.info(f'Model weights saved at: model.pt')

    x_train = data[0]
    y_train = data[1]
    x_test = data[-2]
    y_test = data[-1]

    train_pred = model(x_train.to(device))[:,0]
    thresh_pred = torch.where(train_pred < 0.5, torch.tensor(0), torch.tensor(1))
    zero_mask = (thresh_pred-y_train.to(device) == 0.0)
    train_acc = zero_mask.sum().item()/len(y_train)

    test_pred = model(x_test.to(device))[:,0]
    thresh_pred = torch.where(test_pred < 0.5, torch.tensor(0), torch.tensor(1))
    zero_mask = (thresh_pred-y_test.to(device) == 0.0)
    test_acc = zero_mask.sum().item()/len(y_test)

    logger.info(f'Training accuracy: {train_acc}')
    logger.info(f'Testing accuracy: {test_acc}')

    epoch=0
    feat_thresh=[0.1,0.2,0.3]
    for t in feat_thresh:
        logger.info(f'feat_thresh: {t}')
        epoch=0
        for f in features_train:
            logger.info(f'epoch: {epoch}')
            logger.info(f'alignment: {feature_stats(f.cpu(),data_dim=data_config.dim_in,tree_depth=data_config.depth,dim_in=data_config.dim_in,threshold=t)}')
            epoch+=log_config.log_weight
    


    # Plot the training losses
    plt.plot(Train_losses)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('train_losses.png')
    plt.close()
    logger.info('Training losses plot: train_losses.png')

    # Plot the accuracies in the same plot
    plt.plot(acc_dict['train'], label='Training Accuracy')
    plt.plot(acc_dict['int'], label='Interior Accuracy')
    plt.plot(acc_dict['bound'], label='Boundary Accuracy')
    plt.plot(acc_dict['test'], label='Test Accuracy')
    plt.title('Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracies.png')
    plt.close()
    logger.info('Accuracies plot: accuracies.png')


    log_path_str = log_config.log_path+'/'+str(data_config.depth)+'_'+str(data_config.dim_in)+'_'+str(data_config.num_points)+'.txt'
    # log_path = os.path.join(os.getcwd(),log_path_str)
    
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    
    
        

if __name__ == "__main__":
    main()


    # Run your experiment with the current configuration
    

