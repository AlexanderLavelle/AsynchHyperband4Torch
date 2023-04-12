from cytoolz import valmap
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


N_FEATURES=1


def get_train_val_dicts(dataset):
    dataset.dropna(how='all', axis=1, inplace=True)
    dataset.dropna(how='any', axis=1, inplace=True)
    assert dataset.isna().sum().sum() == 0, 'has NaNs'

    dataset.reset_index(drop=True, inplace=True)

    train = dict(dataset.iloc[:,:-5])
    val = dict(dataset.iloc[:, -5:])

    train = extract_target(train)
    val = extract_target(val)

    return train, val


def extract_target(data):
    data = valmap(lambda x: pd.DataFrame(x), data)
    data = valmap(lambda stock_prices: append_task(stock_prices), data)
    return data


def append_task(stock_prices):
    target = (stock_prices[5:].values > stock_prices[:-5].values).astype(int)
    stock_prices = stock_prices.iloc[:-5].copy()
    stock_prices['target'] = target
    return stock_prices.dropna()


class WindowedDataset(torch.utils.data.Dataset):
    def __init__(self, data, config, stride=None):
        super().__init__()
        self.examples = []
        self.WINDOW_SIZE = config['WINDOW_SIZE']
        self.stride = int(self.WINDOW_SIZE * .85) if stride is None else stride

        self.scaler = MinMaxScaler(feature_range=(0.01,1.))
        
        self.splitter(data)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    def splitter(self, data):
        for k,v in data.items():
            max_idx = len(data[k])
            start = np.arange(start=0, stop=max_idx, step=self.stride)
            end = start + self.WINDOW_SIZE
            if end[-1] > max_idx:
                end[-1] = max_idx
                start[-1] = end[-1] - self.WINDOW_SIZE
                
            for i in range(len(start)):
                sliced = data[k][start[i]:end[i]]
                if (sliced.shape[0] == self.WINDOW_SIZE) & (sliced.shape[1] == N_FEATURES+1):
                    sliced = sliced.copy()
                    sliced[k] = (self.scaler.fit_transform(sliced[k].values.reshape(-1,1)) * 100).astype(int)
                    self.examples.append(sliced)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        
        ex = self.examples[idx]
        
        idx = torch.tensor(ex.index.to_list()).long().to(self.device)
        
        arr = ex.values
        x = torch.tensor(arr[:,:-1]).long().to(self.device)
        y = torch.tensor(arr[:,-1]).float().to(self.device)

        assert x.size()[0] == idx.size()[0]
                
        return x, idx, y 
    

def loader(d, config, batch_size, val=True):
    SHUFFLE = False if val else True
    ds = WindowedDataset(d, config)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=SHUFFLE, 
        drop_last=False
    ) 
    return dl
