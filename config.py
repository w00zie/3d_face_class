"""Hyperparameters for the training session.
"""

config = {'BATCH_SIZE' : 16,
          'NUM_EPOCHS' : 300,
          'LEARNING_RATE': 1e-4,
          'STEP_DECAY': 100,
          'GAMMA_DECAY': 0.9, 
          'DATA_DIR': 'bosphorus_full', # Must be a valid sub-directory of ./graph_data/
          'RANDOM_SEED': 23,
          'NUM_SPLITS': 3, # k value of k-fold cross-validation
          'FILTER_LOWER_THAN': 10, # Will be used only identities with >= number of meshes
          'FILTER_GREATER_THAN': 34,  # Will be used only identities with <= number of meshes
          'DEVICE': 'cuda:0',
          'TOT_TRAIN_PCTG': 0.7, # Total training data percentage (i.e. train/test split)
          'FOLD_TRAIN_PCTG': 0.67, # Training data percentage for every fold in k-cross-validation
          'NET_TYPE': "pool2", # Model type (see `./models.py`)
          'NORM_VERT': True, # Normalize the nodes positions to [0,1]^3
          'USE_COO': True} # Use the COO sparse format for edges
