import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import json
import os
from data import get_train_test_split
from utils import get_model
import argparse

# Parsing ---------------------------------------------------------------------
# Parse the arguments, i.e. the directory containing the model you
# want to test.
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str)
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()
PATH = args.exp_dir
print(f'\nTesting {PATH}\n')
MODEL_PATH = os.path.join(PATH, 'model', 'state_dict.pth')

# Config ----------------------------------------------------------------------
with open(os.path.join(PATH, "config.json")) as json_file:
        config = json.load(json_file)

# Train/test split ------------------------------------------------------------
with open(os.path.join(PATH, "train_test_split.json")) as tts_file:
        train_test_split = json.load(tts_file)

# Classes dict ----------------------------------------------------------------
with open(os.path.join(PATH, "classes_dict.json")) as cd_file:
        classes_dict = json.load(cd_file)

# Device ----------------------------------------------------------------------
device = config["DEVICE"]

# Data ------------------------------------------------------------------------
train_list, test_list = get_train_test_split(config=config,
                                             train_test_split=train_test_split,
                                             classes_dict=classes_dict)
classes_train = torch.stack([tr.y for tr in train_list])
classes_test = torch.stack([te.y for te in test_list])
num_classes = len(torch.unique(torch.cat([classes_train, classes_test])))
print('[DATA] - There are {} training examples, {} testing examples and {} classes'.
    format(len(train_list), len(test_list), num_classes))
# Model -------------------------------------------------------------------------
net = get_model(name=config["NET_TYPE"],
                device=device,
                num_classes=num_classes)
net.load_state_dict(torch.load(MODEL_PATH))
train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('\n[MODEL] - There are {} trainable params in the model\n'.\
    format(train_params))
net.eval()
# --------------------------------------------------------------------------------
train_loader = DataLoader(train_list,
                          batch_size=config['BATCH_SIZE'],
                          shuffle=False, num_workers=4)

test_loader = DataLoader(test_list,
                         batch_size=config['BATCH_SIZE'],
                         shuffle=False, num_workers=4)
wrong_preds_train, wrong_preds_test = [], []
with torch.no_grad():
    train_loss, train_acc = 0., 0.
    test_loss, test_acc = 0., 0.
    net.eval()
    # =========================================================================
    # Train -------------------------------------------------------------------
    for num_batch, data in enumerate(tqdm(train_loader,
                                          desc='Train',
                                          ncols=100)):
        out = net(data.to(device))
        loss = F.nll_loss(out, data.y)
        # Accumulate loss and accuracy
        train_loss += loss.item()
        pred = torch.argmax(out, dim=1)
        eqs = pred.eq(data.y)
        wrong_pred = torch.where(eqs == False)[0]
        if len(wrong_pred) > 0:
            wrong_preds_train.append(num_batch * config['BATCH_SIZE'] + wrong_pred)
        train_acc += eqs.sum().item() / data.num_graphs
    # Test --------------------------------------------------------------------
    for num_batch, data in enumerate(tqdm(test_loader,
                                          desc='Test',
                                          ncols=100)):
        out = net(data.to(device))
        loss = F.nll_loss(out, data.y)
        # Accumulate loss and accuracy
        test_loss += loss.item()
        pred = torch.argmax(out, dim=1)
        eqs = pred.eq(data.y)
        wrong_pred = torch.where(eqs == False)[0]
        if len(wrong_pred) > 0:
            wrong_preds_test.append(num_batch * config['BATCH_SIZE'] + wrong_pred)
        test_acc += eqs.sum().item() / data.num_graphs
    # =========================================================================
    # Calculate metrics -------------------------------------------------------
    train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_acc / len(train_loader)
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_acc / len(test_loader)
    print(f'\n[Train accuracy] = {train_acc}')
    print(f'[Test accuracy] = {test_acc}\n')
    wrong_preds_train = torch.cat(wrong_preds_train)
    wrong_preds_test = torch.cat(wrong_preds_test)

    if args.verbose:
        print('[Train predictions]')
        for num_batch, data in enumerate(train_loader):
            data = data.to_data_list()
            for i in range(config['BATCH_SIZE']):
                elem = num_batch * config['BATCH_SIZE'] + i
                if elem in wrong_preds_train:
                    graph = data[i]
                    idnty = graph.identity
                    print(f'Wrong prediction of {idnty}')
        print('\n[Test predictions]')
        for num_batch, data in enumerate(test_loader):
            data = data.to_data_list()
            for i in range(config['BATCH_SIZE']):
                elem = num_batch * config['BATCH_SIZE'] + i
                if elem in wrong_preds_test:
                    graph = data[i]
                    idnty = graph.identity
                    print(f'Wrong prediction of {idnty}')

