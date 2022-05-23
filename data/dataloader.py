import torch
from torch.utils import data
import os, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train_data',
                 dtype=torch.float32):

        super(LoadData, self).__init__()

        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = torch.load(os.path.join(dataset_dir, ds_name+'.pt'))

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]

    def __getitem__(self, idx):
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        return data_out

if __name__=='__main__':

    data_path = 'datasets_parsed'
    ds = LoadData(data_path)

    dataloader = data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=8, drop_last=True)

    s = time.time()
    for i in range(320):
        a = ds[i]
    print(time.time() - s)
    print('pass')

    dl = iter(dataloader)
    s = time.time()
    for i in range(10):
        a = next(dl)
    print(time.time() - s)
    print('pass')