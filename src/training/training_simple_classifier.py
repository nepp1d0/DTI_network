from time import time
import copy
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss

from src.model.classifier import Classifier
from src.data_processing.trial_loader import trialDataset
from src.model.classifier import Classifier
from src.utils.metrics import roc_curve_, prauc_curve_

# Load data
def load_data(dataset_path, train_val_split = 0.9):
    print('Loading data...')
    dataset = torch.load(dataset_path)
    tot_len = len(dataset)
    train_len = int(train_val_split * tot_len)
    val_len = tot_len - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    print('Data Loaders ready')
    return DataLoader(train_set, shuffle=True, batch_size=32), DataLoader(val_set, shuffle=True, batch_size=32)

def evaluate(data_generator, model, device):
    result_folder = '/home/nick/PycharmProjects/DTI_network/src/Results'
    y_pred = []
    y_label = []
    model.eval()
    for i, (inputs, labels) in enumerate(data_generator):
        labels = Variable(torch.from_numpy(np.array(labels)).float()).to(device)
        inputs = inputs.to(device)
        score = model(inputs)
        act = torch.nn.Sigmoid()
        logits = torch.squeeze(act(score)).detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    model.train()
    ## ROC-AUC curve
    roc_auc_file = os.path.join(result_folder, "roc-auc.jpg")
    plt.figure(0)
    roc_curve_(y_pred, y_label, roc_auc_file)
    plt.figure(1)
    pr_auc_file = os.path.join(result_folder, "pr-auc.jpg")
    prauc_curve_(y_pred, y_label, pr_auc_file)
    return roc_auc_score(y_label, y_pred), \
           average_precision_score(y_label, y_pred), \
           f1_score(y_label, outputs), \
           log_loss(y_label, outputs), y_pred

def train(train_loader, val_loader=None, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model and hyperparameters
    if 'learning_rate' in kwargs.keys():
        learning_rate = kwargs['learning_rate']
    else:
        learning_rate = 1e-4

    if 'decay' in kwargs.keys():
        decay = kwargs['decay']

    if 'batch_size' in kwargs.keys():
        batch_size = kwargs['batch_size']
    else:
        batch_size = 32

    if 'train_epoch' in kwargs.keys():
        train_epoch = kwargs['train_epoch']
    else:
        train_epoch = 50
    verbose = True
    hidden_layers = [1024, 1]
    model = Classifier(input_size=1792, hidden_layers= hidden_layers)
    model = model.to(device)

    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    valid_metric_record = []
    valid_metric_header = ["# epoch"]

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    iteration_loss = 0
    loss_fct = torch.nn.BCELoss()
    t_start = time()
    float2str = lambda x: '%0.4f' % x
    # Training loop
    print('Start training')
    for epoch in range(train_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = Variable(torch.from_numpy(np.array(labels)).float()).to(device)
            inputs = inputs.to(device)

            output = model(inputs)

            act = torch.nn.Sigmoid()
            score = torch.squeeze(act(output), 1)
            score = torch.squeeze(score, 1)
            loss = loss_fct(score, labels)
            loss_history.append(loss.item())
            iteration_loss += 1
            # zero the parameter gradients
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print statistics
            if (i % 100 == 0):
                t_now = time()
                print('Training at Epoch ' + str(epoch + 1) + ' iteration ' + str(i) + \
                      ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                      ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

        if val_loader is not None:
            # validate, select the best model up to now
            with torch.set_grad_enabled(False):
                # binary: ROC-AUC, PR-AUC, F1, cross-entropy loss
                auc, auprc, f1, loss, logits = evaluate(val_loader, model, device)
                lst = ["epoch " + str(epoch)] + list(map(float2str, [auc, auprc, f1]))
                valid_metric_record.append(lst)
                if auc > max_auc:
                    model_max = copy.deepcopy(model)
                    max_auc = auc
                if verbose:
                    print('Validation at Epoch ' + str(epoch + 1) + ', AUROC: ' + str(auc)[:7] + \
                          ' , AUPRC: ' + str(auprc)[:7] + ' , F1: ' + str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                          str(loss)[:7])
        else:
            model_max = copy.deepcopy(model)

    # load early stopped model
    model = model_max
    print('Finished Training')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_path = '/home/nick/PycharmProjects/DTI_network/src/Data/GPCR_train_embeddings.pt'
    train_loader, val_loader = load_data(dataset_path)
    train(train_loader, val_loader)