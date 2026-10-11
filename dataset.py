import torch 
from torch.utils.data import Dataset
from utils.tools import pad_traj_tensor_to_fix_length, get_normalized_padded_traj, get_padded_mask

class ConvTrajDatset(Dataset):
    def __init__(self, traj_dataset, max_length, mean_lon, mean_lat, std_lon, std_lat):
        super(ConvTrajDatset, self).__init__()
        self.mean_lon, self.mean_lat, self.std_lon, self.std_lat = mean_lon, mean_lat, std_lon, std_lat
        self.traj_coor = traj_dataset[0]
        self.traj_lens = traj_dataset[2]
        self.traj_images = traj_dataset[6]
        self.dataset_num = len(self.traj_coor)
        self.traj_ids = torch.arange(0, self.dataset_num).int()
        #padding and mask
        self.padded_tarjs_coor = pad_traj_tensor_to_fix_length(self.traj_coor, max_length, pad_value=0.0)  # shape: [self.dataset_num, max_length, 2]
        self.trajs_mask = get_padded_mask(self.padded_tarjs_coor)  # shape: [self.datset_num, max_length]
        self.padded_normalized_trajs_coor =get_normalized_padded_traj(self.padded_tarjs_coor, self.trajs_mask,mean_lon, mean_lat, std_lon, std_lat)  # shape: [self.dataset_num, max_length, 2]

    def __len__(self):
        return self.dataset_num

    def __getitem__(self, idx):  # shape: [], [max_length, 1], [max_length, 1], [x_grid_num, y_grid_num] # [max_length]
        return self.traj_ids[idx], self.padded_normalized_trajs_coor[idx, :, 0:1], self.padded_normalized_trajs_coor[idx, :, 1:2], self.traj_images[idx], # self.trajs_mask[idx]


class TrajCLDataset(Dataset):
    def __init__(self, traj_dataset, mean_lon, mean_lat, std_lon, std_lat):
        super(TrajCLDataset, self).__init__()
        self.mean_lon, self.mean_lat, self.std_lon, self.std_lat = mean_lon, mean_lat, std_lon, std_lat
        self.traj_coor = traj_dataset[0]
        self.traj_lens = traj_dataset[2]
        # self.traj_x_grid_ids = traj_dataset[3]
        # self.traj_y_grid_ids = traj_dataset[4]
        self.traj_grid_ids = traj_dataset[5]
        self.dataset_num = len(self.traj_coor)
        self.traj_ids = torch.arange(0, self.dataset_num).int()
        # self.traj_soft_grid_ids
        # self.traj_soft_grid_weights
        
    def __len__(self):
        return self.dataset_num
    
    def __getitem__(self, idx):
        coors = self.traj_coor[idx].float()
        grid_ids = torch.as_tensor(self.traj_grid_ids[idx], dtype=torch.float32, device=coors.device).unsqueeze(1)
        grid_xy = torch.cat([grid_ids, coors], dim=1)# shape: [L, 3]，每行是 (grid_id, x, y)

        return self.traj_ids[idx], grid_xy, self.traj_lens[idx]

class TrajSimLeaDataset(Dataset):
    def __init__(self, traj_dataset, mean_lon, mean_lat, std_lon, std_lat):
        super(TrajSimLeaDataset, self).__init__()
        self.mean_lon, self.mean_lat, self.std_lon, self.std_lat = mean_lon, mean_lat, std_lon, std_lat
        self.traj_coor = traj_dataset[0]
        self.dataset_num = len(self.traj_coor)
        self.traj_ids = torch.arange(0, self.dataset_num).int()

    def __len__(self):
        return self.dataset_num
    
    def __getitem__(self, idx):
        return self.traj_coor[idx], self.traj_ids[idx]

class TrajDualSimformer(Dataset):
    def __init__(self, traj_dataset):
        self.traj_coor = traj_dataset[0]
        self.dataset_num = len(self.traj_coor)
        self.traj_ids = torch.arange(0, self.dataset_num).int()
        self.traj_grid_ids = traj_dataset[5]
        
    def __len__(self):
        return self.dataset_num
    
    def __getitem__(self, idx):
        
        coors = self.traj_coor[idx].float()
        grid_ids = torch.as_tensor(self.traj_grid_ids[idx], dtype=torch.float32, device=coors.device).unsqueeze(1)
        grid_xy = torch.cat([grid_ids, coors], dim=1)# shape: [L, 3]，每行是 (grid_id, x, y)
        
        return self.traj_ids[idx], grid_xy