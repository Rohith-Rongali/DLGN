import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model,loss_fn,optimizer,train_dataloader,
            data,log_features_fn,
            log_config,
            train_config,
            print_std=False):
    """
    Trains the given model using the specified loss function and optimizer for the specified number of epochs.
    For now works for DLGN and DNN kind of things where there is no alternating optimization.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        num_epochs (int): The number of epochs to train the model for.
        loss_fn (callable): The loss function to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_dataloader (torch.utils.data.DataLoader): The data loader for the training data.
        data (list): A list containing the training, validation, and test data.
        log_features_fn (callable): A function that takes in a model and returns a list of features to log.
        log_epochs (int, optional): The number of epochs between each logging of loss and print loss and epoch. Defaults to 10.
        log_weight (int, optional): The number of epochs between each logging of features. Defaults to 10.
        log_acc (int, optional): The number of epochs between each logging of accuracy. Set to 0 to disable. Defaults to 0.
        thresh (float, optional): The threshold for early stopping based on loss. Set to 0 to disable. Defaults to 0.01.
        print_std (bool, optional): Whether to print standard output(loss) during training. Defaults to False.

    Returns:
        tuple: A tuple containing the training losses, the logged features, and the accuracy dictionary.
    """
    [x_train,y_train,x_int,y_int,x_bound,y_bound,x_bound_root,y_bound_root,x_test,y_test]=data
    #   features_initial = log_features_fn(model)
    features_train=[]
    model.to(device)
    Train_losses=[]
    acc_dict = {'train':[],'test':[],'int':[],'bound':[],'bound_root':[]}
    for epoch in range(train_config.num_epochs):
        model.train()
        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        if epoch % log_config.log_weight == 0:
            features_train.append(log_features_fn(model))
        if epoch % log_config.log_epochs == 0:
            loss_full = loss_fn(model(x_train.to(device))[:,0],y_train.to(device))
            Train_losses.append(loss_full.item())
            if print_std:
                print(f'Epoch {epoch} Loss {loss_full.item():.4f}')
        if log_config.log_acc!=0:
            if epoch % log_config.log_acc == 0:
                train_pred = model(x_train.to(device))[:,0]
                thresh_pred = torch.where(train_pred < 0.5, torch.tensor(0), torch.tensor(1))
                zero_mask = (thresh_pred-y_train.to(device) == 0.0)
                train_acc = zero_mask.sum().item()/len(y_train)
                acc_dict['train'].append(train_acc)

                int_pred = model(x_int.to(device))[:,0]
                thresh_pred = torch.where(int_pred < 0.5, torch.tensor(0), torch.tensor(1))
                zero_mask = (thresh_pred-y_int.to(device) == 0.0)
                int_acc = zero_mask.sum().item()/len(y_int)
                acc_dict['int'].append(int_acc)

                bound_pred = model(x_bound.to(device))[:,0]
                thresh_pred = torch.where(bound_pred < 0.5, torch.tensor(0), torch.tensor(1))
                zero_mask = (thresh_pred-y_bound.to(device) == 0.0)
                bound_acc = zero_mask.sum().item()/len(y_bound)
                acc_dict['bound'].append(bound_acc)

                test_pred = model(x_test.to(device))[:,0]
                thresh_pred = torch.where(test_pred < 0.5, torch.tensor(0), torch.tensor(1))
                zero_mask = (thresh_pred-y_test.to(device) == 0.0)
                test_acc = zero_mask.sum().item()/len(y_test)
                acc_dict['test'].append(test_acc)

                bound_root_pred = model(x_bound_root.to(device))[:,0]
                thresh_pred = torch.where(bound_root_pred < 0.5, torch.tensor(0), torch.tensor(1))
                zero_mask = (thresh_pred-y_bound_root.to(device) == 0.0)
                bound_root_acc = zero_mask.sum().item()/len(y_bound_root)
                acc_dict['bound_root'].append(bound_root_acc)
                # if print_std:
                #     print(f'Epoch {epoch} train_acc {train_acc} test_acc {test_acc}  // interior {int_acc} boundary {bound_acc} root_node {bound_root_acc}')
        if loss_full.item() < train_config.thresh:
            print(f'Early stopping at epoch {epoch} because loss is below {train_config.thresh}')
            features_train.append(log_features_fn(model))
            break

    # features_final = log_features_fn(model)
    return Train_losses,features_train,acc_dict
