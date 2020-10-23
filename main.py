import torch
from data import get_data
from utils import get_model
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import DataLoader
from config import config

device = config["DEVICE"]

# Data ------------------------------------------------------------------------
train_list, classes_train, test_list, classes_test, classes_dict = \
    get_data(config, return_classes_dict=True)
num_classes = len(torch.unique(torch.stack(classes_train)))
train_test_split = {}
train_test_split['TRAIN_MESHES'] = [tr.identity for tr in train_list]
train_test_split['TEST_MESHES'] = [te.identity for te in test_list]
# Model -----------------------------------------------------------------------
net = get_model(name=config["NET_TYPE"],
                device=device,
                num_classes=num_classes)
train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('\n[MODEL] - There are {} trainable params in the model'.\
    format(train_params))
# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(net.parameters(), lr=config['LEARNING_RATE'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=config['STEP_DECAY'],
                                            gamma=config['GAMMA_DECAY'])
# Organize Tensorboard logging ------------------------------------------------
logdir = 'runs/k_fold_{}_{}'.format(datetime.now().strftime('%b%d_%H-%M-%S_yoda'),
                                    config['DATA_DIR'])
writer = SummaryWriter(logdir)
# Saving hyperparameters
with open(os.path.join(logdir,"config.json"), "w") as f:
            json.dump(config, f)
# Saving train/test split filenames
with open(os.path.join(logdir, "train_test_split.json"), "w") as tts:
    json.dump(train_test_split, tts, indent=4)
# Saving classes dictionary
with open(os.path.join(logdir, "classes_dict.json"), "w") as cd:
    json.dump(classes_dict, cd, indent=4)
# Dumping hparams to Tensorboard
writer.add_text("Hyperparams", json.dumps(config), global_step=0)
# and class distributions
writer.add_histogram('Class Distribution/Train',
                     torch.stack(classes_train),
                     global_step=0)
writer.add_histogram('Class Distribution/Test',
                     torch.stack(classes_test),
                     global_step=0)
os.mkdir(os.path.join(logdir, 'model'))
# -----------------------------------------------------------------------------

kfold = StratifiedShuffleSplit(n_splits=config['NUM_SPLITS'],
                               train_size=config['FOLD_TRAIN_PCTG'],
                               random_state=config['RANDOM_SEED'])
num_workers = 4
best_test_acc = 0.
for epoch in range(config['NUM_EPOCHS']):

    epoch_train_loss, epoch_train_acc = [], []
    epoch_val_loss, epoch_val_acc = [], []

    for num_fold, (train_idx, val_idx) in enumerate(kfold.split(train_list,
                                                                classes_train)):
        train_loader = DataLoader([train_list[i] for i in train_idx],
                                  batch_size=config['BATCH_SIZE'],
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader([train_list[i] for i in val_idx],
                                batch_size=config['BATCH_SIZE'], shuffle=False,
                                num_workers=num_workers)
        train_loss, train_acc = 0., 0.
        net.train()
#        for data in tqdm(train_loader, desc='Train',
#                         ncols=100, position=0, leave=True):
        for data in train_loader:
            # Run an optimization step
            optimizer.zero_grad()
            out = net(data.to(device))
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            # Accumulate loss and accuracy
            with torch.no_grad():
                train_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                train_acc += pred.eq(data.y).sum().item() / data.num_graphs
        # Store the training metrics for every fold
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_acc / len(train_loader)
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        # Validation step -------------------------------------------------------
        # No need for gradients
        with torch.no_grad():
            val_loss, val_acc = 0., 0.
            net.eval()
#            for data in tqdm(val_loader, desc='Validation', ncols=100):
            for data in val_loader:
                out = net(data.to(device))
                loss = F.nll_loss(out, data.y)
                # Accumulate loss and accuracy
                val_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                val_acc += pred.eq(data.y).sum().item() / data.num_graphs
            # Store the validation metrics for every fold
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_acc / len(val_loader)
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)
        # -----------------------------------------------------------------------
        if epoch == 0:
            # Monitor the classes distribution of every fold
            writer.add_histogram('Class Distribution/Train/Fold {}'.\
                format(num_fold),
                       torch.cat([data.y for data in train_loader]),
                       global_step=0)
            writer.add_histogram('Class Distribution/Train/Validation/Fold {}'.\
                format(num_fold),
                       torch.cat([data.y for data in val_loader]),
                       global_step=0)

    with torch.no_grad():
        if epoch % 10 == 0:
            # Log the model weights for visualization
            for name, param in net.named_parameters():
                if param.requires_grad:
                    writer.add_histogram('Model/{}'.format(name), param, epoch)
        # Average the train / validation metrics
        train_loss = torch.mean(torch.tensor(epoch_train_loss))
        train_acc = torch.mean(torch.tensor(epoch_train_acc))
        val_loss = torch.mean(torch.tensor(epoch_val_loss))
        val_acc = torch.mean(torch.tensor(epoch_val_acc))

        if ((epoch+1) % 100 == 0):
            print(f'\nEpoch {epoch} - Saving model...')
            model_path = os.path.join(logdir, 'model', 'state_dict.pth')
            opt_path = os.path.join(logdir, 'model', 'opt_state.pth')
            torch.save(net.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
            print('Done.')
    # ---------------------------------------------------------------------------
    # Log metrics
    writer.add_scalar('Train/Accuracy', train_acc, epoch)
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Validation/Accuracy', val_acc, epoch)
    writer.add_scalar('Validation/Loss', val_loss, epoch)
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    # ---------------------------------------------------------------------------
    print('\n[{:03d}/{:03d}] LOSS--------------   ACCURACY'.\
        format(epoch, config['NUM_EPOCHS']))
    print('[TRAIN]   {}   {} %'.format(train_loss, train_acc))
    print('[VAL]     {}   {} %'.format(val_loss, val_acc))

    # Adjust learning rate
    scheduler.step()

