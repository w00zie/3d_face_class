from torch_geometric.data import Data, DataLoader
import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from scipy.sparse import coo_matrix
from typing import Tuple, List, Dict

def extract_data(data_dir: str,
                 use_coo: bool,
                 filter_lower_than: int,
                 filter_greater_than: int,
                 normalize_v: bool) -> Tuple[List, Dict]:

    """This method extracts the data inside `./graph_data/data_dir/` and returns
    a list of elements manageable by torch_geometric (i.e. a list of torch_geometric.data.Data objects)
    
    Parameters
    ==========
    data_dir: directory of the dataset (e.g. `./graph_data/frgc`)
    use_coo: boolean flag indicating whether to use the COO sparse format on the edges matrix
    filter_lower_than: will filter out identities with a number of meshes lower than this threshold
    filter_greater_than: will filter out identities with a number of meshes greater than this threshold
    normalize_v: boolean flag indicating whether to use the normalization on the nodes positions

    Returns
    =======
    data_list: List[torch_geometric.data.Data]
        List of the dataset elements in a format understood by torch/torch_geometric
    classes: Dict
        Dictionary containing the association `identity : class_id`
    """
    # List where data will be finally stored
    data_list = []
    # Check dataset
    if 'frgc' in data_dir:
        prefix = '0'
    elif 'bosphorus' in data_dir:
        prefix = 'bs'
    else:
        print('Error! Dataset unknown! Possible choices are `frgc` or `bosphorus_full`')
        exit()
    # Setting the path
    data_dir = os.path.join('graph_data', data_dir)
    # Listing all the sub-directories (classes)
    subdirs = [d for d in os.listdir(data_dir) if d.startswith(prefix)]
    print(f'[PRE] There are {len(subdirs)} classes in this dataset')
    # Filtering out identities that have < `filter_lower_than` meshes:
    # Since every mesh is split into separate `nodes`,`edges` and `faces` files then the number of
    # meshes located in a sub-directory must be divided by 3.
    if filter_lower_than > 0:
        subdirs = [d for d in subdirs if int(len(os.listdir(os.path.join(data_dir, d))) / 3) >= filter_lower_than]
        print(f'[PRE] There are {len(subdirs)} classes in this dataset after removing identities with < {filter_lower_than} faces')
    # Filtering out identities that have > `filter_greater_than` meshes:
    if filter_greater_than > 0:
        subdirs = [d for d in subdirs if int(len(os.listdir(os.path.join(data_dir, d))) / 3) <= filter_greater_than]
        print(f'[PRE] There are {len(subdirs)} classes in this dataset after removing identities with > {filter_greater_than} faces')
    # Final classes dictionary
    classes = {subdirs[i]:i for i in range(len(subdirs))}
    # Gathering the data
    for num_sd, sd in enumerate(tqdm(subdirs, desc='Data', ncols=100)):
        # Getting nodes and edges for every mesh
        edges_fn = [f for f in os.listdir(os.path.join(data_dir, sd)) if 'edges' in f]
        nodes_fn = [f for f in os.listdir(os.path.join(data_dir, sd)) if 'nodes' in f]
        # Sorting them in order to be sure that they are indeed referred to the same identity
        edges_fn, nodes_fn = sorted(edges_fn, key=lambda x: x[5:]), sorted(nodes_fn, key=lambda x: x[5:])
        assert len(edges_fn) == len(nodes_fn), 'Some error in retrieving data'
        for edg, nds in zip(edges_fn, nodes_fn):
            e = np.load(os.path.join(data_dir,sd,edg))
            if not use_coo:
                e = torch.from_numpy(e).type(torch.long)
            else:
                e = torch.from_numpy(coo_matrix(e).toarray()).type(torch.long)
            v = torch.from_numpy(np.load(os.path.join(data_dir, sd, nds))).type(torch.float32)
            if normalize_v:
                minv = torch.min(v)
                scaled_unit = 1. / (torch.max(v) - minv)
                v = scaled_unit * (v - minv)
            tg_data = Data(edge_index=e.T, pos=v, y=torch.tensor(classes[sd]))
            tg_data.identity = edg[6:].split('.')[0]
            data_list.append(tg_data)
    if len(data_list) == 0:
        print('Empty data list! Try lowering the `filter_lower_than` param.')
        exit()
    return (data_list, classes)

def build_emotion_data_list(data_dir: str,
                            use_coo: str,
                            normalize_v: str) -> Tuple[List, Dict]:
    """This method data the emotional data inside `./graph_data/data_dir/` and returns
    a list of elements manageable by torch_geometric (i.e. a list of torch_geometric.data.Data objects).

    This emotional data is extracted by using the information on the filenames: i.e. every filename 
    must contain one of the labels from `ANGER, FEAR, SADNESS, HAPPY, DISGUST, SURPRISE`.

    Parameters
    ==========
    data_dir: directory of the dataset (e.g. `./graph_data/frgc`)
    use_coo: boolean flag indicating whether to use the COO sparse format on the edges matrix
    normalize_v: boolean flag indicating whether to use the normalization on the nodes positions

    Returns
    =======
    data_list: List[torch_geometric.data.Data]
        List of the dataset elements in a format understood by torch/torch_geometric
    classes: Dict
        Dictionary containing the association `identity : class_id`
    """
    data_list = []
    data_dir = os.path.join('graph_data', data_dir)
    subdirs = sorted([d for d in os.listdir(data_dir) if d.startswith('bs')])
    classes = {'ANGER': 0, 'SADNESS': 1, 'HAPPY': 2,
               'DISGUST': 3, 'SURPRISE': 4, 'FEAR': 5}
    association = {}
    edges_fn = []
    nodes_fn = []
    for sd in tqdm(subdirs, desc='Data', ncols=100):
        for emotion in classes.keys():
            for f in os.listdir(os.path.join(data_dir, sd)):
                full_name = os.path.join(data_dir, sd, f)
                if emotion in f and 'edges' in f:
                    edges_fn.append(full_name)
                    association[full_name] = classes[emotion]
                if emotion in f and 'nodes' in f:
                    nodes_fn.append(full_name)
                    association[full_name] = classes[emotion]
    edges_fn, nodes_fn = sorted(edges_fn, key=lambda x: x[-8:]), sorted(nodes_fn, key=lambda x: x[-8:])
    assert len(edges_fn) == len(nodes_fn), 'Some error in retrieving data'
    zipped_fnames = zip(edges_fn, nodes_fn)
    for edg, nds in zipped_fnames:
        e = np.load(edg)
        if not use_coo:
            e = torch.from_numpy(e).type(torch.long)
        else:
            e = torch.from_numpy(coo_matrix(e).toarray()).type(torch.long)
        v = torch.from_numpy(np.load(nds)).type(torch.float32)
        if normalize_v:
            minv = torch.min(v)
            scaled_unit = 1. / (torch.max(v) - minv)
            v = scaled_unit * (v - minv)
        assert association[edg] == association[nds], 'Some error in class association'
        tg_data = Data(edge_index=e.T, pos=v, y=torch.tensor(association[edg]))
        tg_data.identity = edg[5:].split('.')[0]
        data_list.append(tg_data)
    if len(data_list) == 0:
        print('No emotional information in this dataset! Quitting...')
        exit()
    return data_list, classes

def get_data(config: dict, return_classes_dict: bool = False):
    data_list, classes_dict = extract_data(data_dir=config['DATA_DIR'],
                                           use_coo=config['USE_COO'],
                                           filter_lower_than=config['FILTER_LOWER_THAN'],
                                           filter_greater_than=config['FILTER_GREATER_THAN'],
                                           normalize_v=config['NORM_VERT'])

    train_list, classes_train, test_list, classes_test = \
        _build_train_test_split(data_list, config['TOT_TRAIN_PCTG'])
    if return_classes_dict:
        return (train_list, classes_train, test_list, classes_test, classes_dict)
    else:
        return (train_list, classes_train, test_list, classes_test)

def get_emotion_data(config: dict, return_classes_dict: bool = False):
    data_list, classes_dict = build_emotion_data_list(data_dir=config['DATA_DIR'],
                                                      use_coo=config['USE_COO'],
                                                      normalize_v=config['NORM_VERT'])

    train_list, classes_train, test_list, classes_test = \
        _build_train_test_split(data_list, config['TOT_TRAIN_PCTG'])
    if return_classes_dict:
        return (train_list, classes_train, test_list, classes_test, classes_dict)
    else:
        return (train_list, classes_train, test_list, classes_test)

def get_train_test_split(config: dict, train_test_split: dict, classes_dict: dict):
    """This method returns the data specified by a certain train/test split.

    Parameters
    ==========
    config: Configuration containing the hyperparameters information
    train_test_split: Dictionary that specifies the membership of each mesh into one between
    the train set or the test set.
    classes_dict: Dictionary containing the association `identity : class_id`, needed for retrieval

    Return
    ======
    train_list: List
        List containing the training elements, each in a torch_geometric.data.Data format
    test_list: List
        List containing the test elements, each in a torch_geometric.data.Data format
    """
    data_dir = os.path.join('graph_data', config['DATA_DIR'])
    train_ids = train_test_split['TRAIN_MESHES']
    test_ids = train_test_split['TEST_MESHES']
    train_list = _build_data_list(config, data_dir, train_ids, classes_dict)
    test_list = _build_data_list(config, data_dir, test_ids, classes_dict)
    return (train_list, test_list)

def _build_data_list(config: dict,
                     data_dir: str,
                     split_list: list,
                     classes_dict: dict):
    data_list = []
    for elem in split_list:
        idnt = elem[:5]
        e_fname = os.path.join(data_dir, idnt, 'edges_{}.npy'.format(elem))
        n_fname = os.path.join(data_dir, idnt, 'nodes_{}.npy'.format(elem))
        e = np.load(e_fname)
        v = torch.from_numpy(np.load(n_fname)).type(torch.float32)
        if config['USE_COO']:
            e = torch.from_numpy(coo_matrix(e).toarray()).type(torch.long)
        else:
            e = torch.from_numpy(e).type(torch.long)
        if config['NORM_VERT']:
            minv = torch.min(v)
            scaled_unit = 1. / (torch.max(v) - minv)
            v = scaled_unit * (v - minv)
        tg_data = Data(edge_index=e.T, pos=v, y=torch.tensor(classes_dict[idnt]))
        tg_data.identity = elem
        data_list.append(tg_data)
    return data_list

def _get_stratified_shuffle_split(data_list, train_size=0.8, seed=42):
    kfold = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    classes = [data.y for data in data_list]
    splitted_data = kfold.split(data_list, classes)
    splitted_data = list(splitted_data)[0]
    train_idx, test_idx = splitted_data[0], splitted_data[1]
    return train_idx, test_idx

def _build_train_test_split(data_list: List, train_size: float) -> Tuple[List, List, List, List]:

    train_indices, test_indices = _get_stratified_shuffle_split(data_list, train_size=train_size)
    train_list = [data_list[i] for i in train_indices]
    test_list = [data_list[i] for i in test_indices]

    classes = [data.y for data in data_list]
    classes_train = [data.y for data in train_list]
    classes_test = [data.y for data in test_list]

    print('\n[DATA] - There are {} faces in the data set.'.format(len(data_list)))
    print('[DATA] - There are {} faces in the train set and {} in the test set'.\
        format(len(train_list), len(test_list)))
    num_classes_entire = len(torch.unique(torch.stack(classes)))
    num_classes_train = len(torch.unique(torch.stack(classes_train)))
    num_classes_test = len(torch.unique(torch.stack(classes_test)))

    # Assert that every class is present both in the train and test set
    assert num_classes_entire == num_classes_train == num_classes_test
    print(f'[DATA] - There are {num_classes_entire} classes.\n')

    return (train_list, classes_train, test_list, classes_test)

if __name__ == "__main__":

    from time import time

    config = {'DATA_DIR': 'bosphorus_full',
              'USE_COO': True,
              'NORM_VERT': True,
              'TOT_TRAIN_PCTG': 0.7,
              'BATCH_SIZE': 16,
              'FILTER_LOWER_THAN': 10,
              'FILTER_GREATER_THAN': 34,
              'NUM_SPLITS': 5,
              'FOLD_TRAIN_PCTG': 0.8,
              'RANDOM_STATE': 23
              }

    train_list, classes_train, test_list, classes_test = get_data(config)
    kfold = StratifiedShuffleSplit(n_splits=config['NUM_SPLITS'],
                                   train_size=config['FOLD_TRAIN_PCTG'],
                                   random_state=config['RANDOM_STATE'])
    num_workers = 4
    start_tot = time()
    for epoch in range(3):
        start_epoch = time()
        for num_fold, (train_idx, val_idx) in enumerate(kfold.split(train_list,
                                                                    classes_train)):
            train_loader = DataLoader([train_list[i] for i in train_idx],
                                    batch_size=config['BATCH_SIZE'],
                                    shuffle=True, num_workers=num_workers)
            start_iter = time()
            #for _ in train_loader:
            for _ in tqdm(train_loader, desc='Train', ncols=100, position=0, leave=True):
                pass
            print(f'\t\t{num_fold}: Train iteration took {time() - start_iter} s')
            val_loader = DataLoader([train_list[i] for i in val_idx],
                                    batch_size=config['BATCH_SIZE'],
                                    shuffle=False, num_workers=num_workers)
            start_iter = time()
            #for _ in val_loader:
            for _ in tqdm(val_loader, desc='Val', ncols=100, position=0, leave=True):
                pass
            print(f'\t\t{num_fold}: Val iteration took {time() - start_iter} s')
        print(f'\t{epoch}: Epoch took {time() - start_epoch} s')
    print(f'Whole process took {time() - start_tot} s')
