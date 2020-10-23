import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import InstanceNorm, BatchNorm
from torch_geometric.nn import TopKPooling, SAGPooling
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import global_max_pool, global_mean_pool

class PoolNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(PoolNet, self).__init__()

        self.att_w = nn.Sequential(
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        self.att_net = nn.Sequential(
            nn.Linear(256,256),
            nn.ELU()
        )

        self.conv1 = GATConv(3, 16, heads=2)
        self.norm1 = InstanceNorm(32)
        self.pool1 =  TopKPooling(32, ratio=0.3)

        self.conv2 = GATConv(32, 64, heads=2)
        self.norm2 = InstanceNorm(128)
        self.pool2 =  TopKPooling(128, ratio=0.3)

        self.conv3 = GATConv(128, 256, heads=1)
        self.norm3 = InstanceNorm(256)

        self.att = GlobalAttention(gate_nn=self.att_w,
                                   nn=self.att_net)

        self.lin1 = Linear(256, 512)
        self.lin2 = Linear(512, 128)
        self.lin3 = Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch
        #print(f'\nInput = {x.shape}, {edge_index.shape}, {batch.shape}')
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.elu(x)
        #print(f'Conv1 = {x.shape}, {edge_index.shape}, {batch.shape}')
        x, edge_index, _, batch, _, _ = self.pool1(x=x, edge_index=edge_index, batch=batch)
        #print(f'Pool1 = {x.shape}, {edge_index.shape}, {batch.shape}')
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.elu(x)
        #print(f'Conv2 = {x.shape}, {edge_index.shape}, {batch.shape}')
        x, edge_index, _, batch, _, _ = self.pool2(x=x, edge_index=edge_index, batch=batch)
        #print(f'Pool2 = {x.shape}, {edge_index.shape}, {batch.shape}')
        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)
        x = F.elu(x)
        #print(f'Conv3 = {x.shape}, {edge_index.shape}, {batch.shape}')
        ##x = global_max_pool(x, batch)
        x = self.att(x, batch)
        #print(f'MAXPOOL = {x.shape}, {edge_index.shape}, {batch.shape}')
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.lin1(x))
        #print(f'Lin1 = {x.shape}')
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.lin2(x))
        #print(f'Lin2 = {x.shape}')
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        #print(f'Lin3 = {x.shape}')
        return F.log_softmax(x, dim=1)

class PoolNetv2(torch.nn.Module):
    def __init__(self, num_classes):
        super(PoolNetv2, self).__init__()

        self.att_w = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

        self.att_net = nn.Sequential(
            nn.Linear(512, 512),
            #nn.Tanh()
            nn.ELU()
        )

        self.conv1 = GATConv(3, 32, heads=2)
        self.norm1 = InstanceNorm(64, affine=True)
        self.pool1 =  TopKPooling(64, ratio=0.3, nonlinearity=torch.sigmoid)

        self.conv2 = GATConv(64, 128, heads=2)
        self.norm2 = InstanceNorm(256, affine=True)
        self.pool2 =  TopKPooling(256, ratio=0.3, nonlinearity=torch.sigmoid)

        self.conv3 = GATConv(256, 512, heads=2, concat=False)
        self.norm3 = InstanceNorm(512, affine=True)

        self.att = GlobalAttention(gate_nn=self.att_w,
                                   nn=self.att_net)

        self.lin1 = Linear(512, 512)
        self.lin2 = Linear(512, 256)
        self.lin3 = Linear(256, num_classes)

    def forward(self, data):
        # Input data
        x, edge_index, batch = data.pos, data.edge_index, data.batch

        # 1: Conv - ELU - Norm - Pool
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.norm1(x, batch)
        x, edge_index, _, batch, _, _ = self.pool1(x=x, edge_index=edge_index, batch=batch)

        # 2: Conv - ELU - Norm - Pool
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.norm2(x, batch)
        x, edge_index, _, batch, _, _ = self.pool2(x=x, edge_index=edge_index, batch=batch)

        # 3: Conv - ELU - Norm
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.norm3(x, batch)

        # Global pooling
        ##x = global_max_pool(x, batch)
        x = self.att(x, batch)

        # Dense layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=1)


class PoolNetv3(torch.nn.Module):
    def __init__(self, num_classes):
        super(PoolNetv3, self).__init__()

        self.att_w = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

        self.att_net = nn.Sequential(
            nn.Linear(256, 256),
            #nn.Tanh()
            nn.ELU()
        )

        self.conv1 = GATConv(3, 32, heads=2)
        self.norm1 = InstanceNorm(64, affine=True)
        self.pool1 =  TopKPooling(64, ratio=0.3, nonlinearity=torch.tanh)

        self.conv2 = GATConv(64, 128, heads=2)
        self.norm2 = InstanceNorm(256, affine=True)

        self.att = GlobalAttention(gate_nn=self.att_w,
                                   nn=self.att_net)

        self.lin1 = Linear(256, 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, num_classes)

    def forward(self, data):
        # Input data
        x, edge_index, batch = data.pos, data.edge_index, data.batch

        # 1: Conv - ELU - Norm - Pool
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.norm1(x, batch)
        x, edge_index, _, batch, _, _ = self.pool1(x=x, edge_index=edge_index, batch=batch)

        # 2: Conv - ELU - Norm - Pool
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.norm2(x, batch)

        # Global pooling
        ##x = global_max_pool(x, batch)
        x = self.att(x, batch)

        # Dense layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":

    from data import get_data
    from torch_geometric.data import DataLoader
    from tqdm.auto import tqdm

    config = {'BATCH_SIZE' : 16,
          'NUM_EPOCHS' : 1600,
          'LEARNING_RATE': 0.001,
          'STEP_DECAY': 1000,
          'GAMMA_DECAY': 0.9,
          'DATA_DIR': 'frgc',
          'RANDOM_SEED': 23,
          'NUM_SPLITS': 3,
          'FILTER_LOWER_THAN': 20,
          'FILTER_GREATER_THAN': -1,
          'DEVICE': 'cuda:0',
          'TOT_TRAIN_PCTG': 0.7,
          'FOLD_TRAIN_PCTG': 0.67,
          'NET_TYPE': "pool2",
          'NORM_VERT': True,
          'USE_COO': True}

    device = config['DEVICE']
    # Data ------------------------------------------------------------------------
    train_list, classes_train, test_list, classes_test = get_data(config)
    num_classes = len(torch.unique(torch.stack(classes_train)))
    net = PoolNet(num_classes=num_classes)

    test_loader = DataLoader(test_list,
                         batch_size=config['BATCH_SIZE'],
                         shuffle=False, num_workers=4)
    acc = 0.
    for i, data in enumerate(tqdm(test_loader, desc='Test', ncols=100)):
        print(16*'-'+str(i)+16*'-')
        out = net(data)
        pred = torch.argmax(out, dim=1)
        acc += pred.eq(data.y).sum().item() / data.num_graphs
        break
    print(100 * acc / len(test_loader))

