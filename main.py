
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
from copy import deepcopy


from data import generate_points,TreeNode,gen_std_basis_DT,CustomDataset

from log_functions import log_features_dlgn,log_features_dlgn_sf,log_features_DLGN_kernel,feature_stats

# from helper import plot_histogram

from models import DLGN,DLGN_SF,DLGN_Kernel,DNN

from kernels import ScaledSig,gate_score,compute_npk,NPK

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can change the following parameters. can 
#Data parameters
depth=4
dim_in = 20
type_data = 'spherical'
num_points = 30000
feat_index_start=0 # Hard-coded
radius=1

#Model parameters
width = 100
depth = 4
beta = 4
bias_fn = True
bias_vn = False

#Training parameters
num_epochs = 400
batch_size = 32
lr = 0.001

#SVM parameters
kernel = 'rbf'
c = 1
gamma = 'scale'



# Now you can use gen_std_basis_data function
X,Y = gen_std_basis_DT(depth, dim_in, type_data, num_points, feat_index_start, radius)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42, stratify=Y
)

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create an SVM classifier with a Gaussian (RBF) kernel
svm_classifier = SVC(kernel=kernel, C=c, gamma=gamma)

# Train the SVM classifier on the training data
svm_classifier.fit(x_train, y_train)
# Make predictions on the test data
y_pred = svm_classifier.predict(x_test)

# Calculate accuracy
accuracy = (y_test == torch.tensor(y_pred)).sum().item()/len(y_test)
print(f"Accuracy: {accuracy * 100}%")


def train(model,num_epochs,loss_fn,optimizer,train_dataloader,data,log_features=10,log_epochs=10,log_weight=10,log_acc=0,thresh=0.01):
  [x_train,y_train,x_int,y_int,x_bound,y_bound,x_bound_root,y_bound_root,x_test,y_test]=data
  features_initial = log_features(model)
  features_train=[]
  model.to(device)
  Train_losses=[]
  for epoch in range(num_epochs):
      model.train()
      for x_batch, y_batch in train_dataloader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)
          pred = model(x_batch)[:, 0]
          loss = loss_fn(pred, y_batch)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()


      if epoch % log_weight == 0:
          features_train.append(log_features(model))
      if epoch % log_epochs == 0:
          loss_full = loss_fn(model(x_train.to(device))[:,0],y_train.to(device))
          Train_losses.append(loss_full.item())
          print(f'Epoch {epoch} Loss {loss_full.item():.4f}')
      if log_acc!=0:
        if epoch % log_acc == 0:
          train_pred = model(x_train.to(device))[:,0]
          thresh_pred = torch.where(train_pred < 0.5, torch.tensor(0), torch.tensor(1))
          zero_mask = (thresh_pred-y_train.to(device) == 0.0)
          train_acc = zero_mask.sum().item()/len(y_train)

          int_pred = model(x_int.to(device))[:,0]
          thresh_pred = torch.where(int_pred < 0.5, torch.tensor(0), torch.tensor(1))
          zero_mask = (thresh_pred-y_int.to(device) == 0.0)
          int_acc = zero_mask.sum().item()/len(y_int)

          bound_pred = model(x_bound.to(device))[:,0]
          thresh_pred = torch.where(bound_pred < 0.5, torch.tensor(0), torch.tensor(1))
          zero_mask = (thresh_pred-y_bound.to(device) == 0.0)
          bound_acc = zero_mask.sum().item()/len(y_bound)

          test_pred = model(x_test.to(device))[:,0]
          thresh_pred = torch.where(test_pred < 0.5, torch.tensor(0), torch.tensor(1))
          zero_mask = (thresh_pred-y_test.to(device) == 0.0)
          test_acc = zero_mask.sum().item()/len(y_test)

          bound_root_pred = model(x_bound_root.to(device))[:,0]
          thresh_pred = torch.where(bound_root_pred < 0.5, torch.tensor(0), torch.tensor(1))
          zero_mask = (thresh_pred-y_bound_root.to(device) == 0.0)
          bound_root_acc = zero_mask.sum().item()/len(y_bound_root)


          print(f'Epoch {epoch} train_acc {train_acc} test_acc {test_acc}  // interior {int_acc} boundary {bound_acc} root_node {bound_root_acc}')
      if loss_full.item() < thresh:
          print(f'Early stopping at epoch {epoch} because loss is below 0.01')
          break

  features_final = log_features(model)
  return Train_losses,features_initial,features_train,features_final
    
