
import torch
from torch import nn
from functools import reduce
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import itertools
import os


from data import generate_points,TreeNode,gen_std_basis_DT,CustomDataset

from log_functions import log_features_dlgn,log_features_dlgn_sf,log_features_DLGN_kernel,feature_stats

# from helper import plot_histogram

# from models import DLGN,DLGN_SF,DLGN_Kernel,DNN

# from kernels import ScaledSig,gate_score,compute_npk,NPK
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# from copy import deepcopy
# from data import generate_points,TreeNode,gen_std_basis_DT,CustomDataset
# from log_functions import log_features_dlgn,log_features_dlgn_sf,log_features_DLGN_kernel,feature_stats
# from models import DLGN,DLGN_SF,DLGN_Kernel,DNN
# from kernels import ScaledSig,gate_score,compute_npk,NPK


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can change the following parameters. can 
#Data parameters
# depth=4
# dim_in = 20
type_data = 'spherical'
# num_points = 30000
# feat_index_start=0 # Hard-coded
# radius=1

# #Model parameters
# width = 100
# depth = 4
# beta = 4
# bias_fn = True
# bias_vn = False

# #Training parameters
# num_epochs = 400
# batch_size = 32
# lr = 0.001

#SVM parameters
# kernel = 'rbf'
# c = 1
# gamma = 'scale'
# degree = 3

# Log path
log_path = 'logs/svm_kernel'


# train_dataset = CustomDataset(x_train, y_train)
# test_dataset = CustomDataset(x_test, y_test)

# # Create DataLoaders
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def svm_acc(svm_classifier,x_train,y_train,x_test,y_test):
        # Train the SVM classifier on the training data
    svm_classifier.fit(x_train.numpy(), y_train.numpy())
    # Make predictions on the test data

    # Calculate train accuracy
    train_accuracy = (y_train == torch.tensor(svm_classifier.predict(x_train))).sum().item()/len(y_train)

    # Calculate test accuracy
    test_accuracy = (y_test == torch.tensor(svm_classifier.predict(x_test))).sum().item()/len(y_test)

    return train_accuracy, test_accuracy


def svm_kernel_runner(depth, dim_in, num_points, kernel, log_path):

    # Now you can use gen_std_basis_data function
    X,Y = gen_std_basis_DT(depth, dim_in, type_data, num_points)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=Y
    )

    C_range = np.logspace(-2, 2, 5)
    gamma = 'scale'
    degrees = np.array([2, 3, 4, 5])

    # if kernel=='linear':
    #     param_grid = dict(C=C_range)
    # elif kernel=='poly':
    #     param_grid = dict(gamma=gamma, C=C_range, degree=degrees)
    # else:
    #     param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    # grid.fit(X.numpy(), Y.numpy())

    # c = grid.best_params_['C']
    # if kernel == 'poly':
    #     degree = grid.best_params_['degree']
    #     gamma = grid.best_params_['gamma']
    # elif kernel == 'rbf':
    #     gamma = grid.best_params_['gamma']
    
    if kernel == 'linear': 
        # write code to find best c based on test accuracy for linear kernel using svm_acc function
        best_c = None
        best_test_acc = 0.0
        best_train_acc = 0.0
        for c in C_range:
            svm_classifier = SVC(kernel=kernel, C=c)
            train_acc, test_acc = svm_acc(svm_classifier, x_train, y_train, x_test, y_test)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_c = c
                best_train_acc = train_acc
        # print(f"Best C for linear kernel: {best_c}")
    elif kernel == 'poly':
        # write code to find best c, degree based on test accuracy for poly kernel using svm_acc function
        best_c = None
        best_degree = None
        best_test_acc = 0.0
        best_train_acc = 0.0
        for c in C_range:
            for degree in degrees:
                svm_classifier = SVC(kernel=kernel, C=c, degree=degree, gamma=gamma)
                train_acc, test_acc = svm_acc(svm_classifier, x_train, y_train, x_test, y_test)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_c = c
                    best_degree = degree
                    best_train_acc = train_acc
        # print(f"Best C for poly kernel: {best_c}")
        # print(f"Best degree for poly kernel: {best_degree}")

    elif kernel == 'rbf':
        # write code to find best c based on test accuracy for rbf kernel using svm_acc function
        best_c = None
        best_test_acc = 0.0
        best_train_acc = 0.0
        for c in C_range:
            svm_classifier = SVC(kernel=kernel, C=c, gamma=gamma)
            train_acc, test_acc = svm_acc(svm_classifier, x_train, y_train, x_test, y_test)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_c = c
                best_train_acc = train_acc
        # print(f"Best C for rbf kernel: {best_c}")


    # Log the results
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, 'acc_log.txt'), 'a') as f:
        f.write(f"Data params: Tree_depth={depth}, dim_in={dim_in}, num_points={num_points}\n")
        if kernel == 'poly':
            f.write(f"SVM params: kernel={kernel}, c={best_c}, gamma = {gamma}, degree={best_degree}\n")
        elif kernel == 'linear':
            f.write(f"SVM params: kernel={kernel}, c={best_c}\n")
        else:
            f.write(f"SVM params: kernel={kernel}, c={best_c}, gamma = {gamma}\n")
        f.write(f"Train accuracy: {best_train_acc}\n")
        f.write(f"Test accuracy: {best_test_acc}\n\n\n")


# call the function for various parameters of depth={3,4,5}, dim_in={15,20,25,30,50}, num_points={10000,20000,30000,40000,50000}, kernel={'linear','rbf','poly'}, c={0.001,0.01,0.1,1,10,100,1000}, degree={2,3,4,5,10}(only for poly)


depths = [3]
dim_ins = [15, 20, 25, 30, 50]
num_points_list = [10000, 20000,25000, 30000, 40000, 50000]
kernels = ['linear', 'rbf', 'poly']

for depth, dim_in, num_points, kernel in itertools.product(depths, dim_ins, num_points_list, kernels):
    print(f"Running for depth={depth}, dim_in={dim_in}, num_points={num_points}, kernel={kernel}")
    svm_kernel_runner(depth, dim_in, num_points, kernel, 'logs/svm_kernel')




