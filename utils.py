from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
seed = 1
import random
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def load_data(path="C:/Users/85433/Desktop/demo/datadata/esca_log.csv"):
    with open(path) as f:
        x = pd.read_csv(f,index_col=0).transpose().astype(np.float32)
    f.close()
    x = np.array(x)
    print('samples', x.shape)
    x = x[:, 2:]
    print(x)
    print('samples', x.shape)
    return x


class GeneDataset(Dataset):

    def __init__(self):
        self.x = load_data()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(idx))
