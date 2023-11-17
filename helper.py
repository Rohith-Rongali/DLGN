import matplotlib.pyplot as plt
from dataclasses import dataclass, replace

from data import generate_points,TreeNode,gen_std_basis_DT,CustomDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import torch

def plot_histogram(data, bins=40):
    # Create the histogram
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram with Statistics')

    # Show the plot
    plt.show()

def return_bound(x_train,y_train):
    knn_1 = KNeighborsClassifier(n_neighbors=5)
    knn_1.fit(x_train,y_train)
    knn_2 = KNeighborsClassifier(n_neighbors=10)
    knn_2.fit(x_train,y_train)
    knn_3 = KNeighborsClassifier(n_neighbors=20)
    knn_3.fit(x_train,y_train)

    y_pred = torch.tensor(knn_1.predict(x_train))
    comp_1 = (y_pred == y_train).int()

    y_pred = torch.tensor(knn_2.predict(x_train))
    comp_2 = (y_pred == y_train).int()

    y_pred = torch.tensor(knn_3.predict(x_train))
    comp_3 = (y_pred == y_train).int()

    int_points = torch.flatten(torch.nonzero(comp_1*comp_2*comp_3)) #intersection of the 3 sets (the points that are easy to classifyby k-NN)

    full =list(range(int(len(x_train))))
    bound = [x for x in full if x not in int_points]

    x_int = x_train[int_points]
    y_int = y_train[int_points]
    x_bound = x_train[bound]
    y_bound = y_train[bound]

    return x_int,y_int,x_bound,y_bound

def return_data_elements(DataConfig, TrainConfig):
    # returns [data] and data_loader

    X,Y = gen_std_basis_DT(depth = DataConfig.depth, dim_in = DataConfig.dim_in, num_points = DataConfig.num_points,type_data= DataConfig.type_data, radius = DataConfig.radius)

    x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42, stratify=Y)

    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=TrainConfig.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TrainConfig.batch_size, shuffle=False)

    x_int,y_int,x_bound,y_bound = return_bound(x_train,y_train)

    bound_list_root=[]
    bound_thresh_root=0.01
    num_nodes = 2**DataConfig.tree_depth-1

    for i in range(len(x_train)):
        if (torch.abs(x_train[i][0])<bound_thresh_root).any():  #if the first coordinate is less than the threshold
            bound_list_root.append(i) #then it is a boundary point
    
    x_bound_root = x_train[bound_list_root]
    y_bound_root = y_train[bound_list_root]

    data = [x_train,y_train,x_int,y_int,x_bound,y_bound,x_bound_root,y_bound_root,x_test,y_test]

    return data, train_dataloader, test_dataloader





    