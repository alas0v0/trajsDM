import torch 
import numpy as np
from torch.utils.data import Dataset
from utils.tools import pad_traj_tensor_to_fix_length, get_normalized_padded_traj, get_padded_mask, get_normalized_padded_traj_convtraj
class ConvTrajDatset(Dataset):
    def __init__(self, traj_dataset, max_length, mean_and_std, range, grid):
        super(ConvTrajDatset, self).__init__()
        self.mean_lon, self.mean_lat, self.std_lon, self.std_lat = mean_and_std
        self.lon_range, self.lat_range = range
        self.grid = grid
        self.traj_coor = traj_dataset[0]
        self.x_grid_list = traj_dataset[3]
        self.y_grid_list = traj_dataset[4]
        self.traj_lens = traj_dataset[2]
        self.dataset_num = len(self.traj_coor)
        self.traj_ids = torch.arange(0, self.dataset_num).int()

        #image origin
        self.traj_images = self.image_encode(self.x_grid_list, self.y_grid_list, self.grid.grid_num_x, self.grid.grid_num_y)
        self.traj_images = torch.tensor(self.traj_images)

        #padding and mask
        self.padded_tarjs_coor = pad_traj_tensor_to_fix_length(self.traj_coor, max_length, pad_value=0.0)  # shape: [self.dataset_num, max_length, 2]
        self.trajs_mask = get_padded_mask(self.padded_tarjs_coor)  # shape: [self.datset_num, max_length]
        #soft image
        self.traj_imags = self.image_encode_soft(self.padded_tarjs_coor, self.trajs_mask, self.grid.grid_num_x, self.grid.grid_num_y, self.grid.temperature, self.grid.delta)
        # self.padded_normalized_trajs_coor =get_normalized_padded_traj(self.padded_tarjs_coor, self.trajs_mask,self.mean_lon, self.mean_lat, self.std_lon, self.std_lat)  # shape: [self.dataset_num, max_length, 2]
        self.padded_normalized_trajs_coor = get_normalized_padded_traj_convtraj(self.padded_tarjs_coor, self.trajs_mask, self.lon_range[0], self.lat_range[0])


    def __len__(self):
        return self.dataset_num

    def __getitem__(self, idx):  # shape: [], [max_length, 1], [max_length, 1], [x_grid_num, y_grid_num] # [max_length]
        return self.traj_ids[idx], self.padded_normalized_trajs_coor[idx, :, 0:1], self.padded_normalized_trajs_coor[idx, :, 1:2], self.traj_images[idx], # self.trajs_mask[idx]

    @staticmethod
    def image_encode(x_list, y_list, x_num_classes, y_num_classes):
        """ 
        硬编码traj image, 返回的格式为嵌套列表
        """
        image_list = []
        for i in range(len(x_list)):
            image = np.zeros((x_num_classes, y_num_classes))
            for j in range(len(x_list[i])):
                image[x_list[i][j]][y_list[i][j]] = 1.0
            image_list.append(image.tolist())
        return image_list
    
    @staticmethod
    def image_encode_soft(points, valid_mask, height, width, temperature, grid_size):
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        x = points[..., 0] / grid_size
        y = points[..., 1] / grid_size
        center_x = torch.floor(x).to(torch.long)
        center_y = torch.floor(y).to(torch.long)
        valid_bool = valid_mask > 0
        temp = max(float(temperature), 1e-12)

        neighbor_offsets = torch.tensor(
            [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
            device=device,
            dtype=torch.long,
        )
        offset_x = neighbor_offsets[:, 0].view(1, 1, -1)
        offset_y = neighbor_offsets[:, 1].view(1, 1, -1)

        ngx = center_x.unsqueeze(-1) + offset_x
        ngy = center_y.unsqueeze(-1) + offset_y
        inside = ( valid_bool.unsqueeze(-1) & (ngx >= 0) & (ngx < height) & (ngy >= 0) & (ngy < width) )

        ndist2 = (x.unsqueeze(-1) - (ngx.to(dtype) + 0.5)) ** 2 + (y.unsqueeze(-1) - (ngy.to(dtype) + 0.5)) ** 2
        # logits = -ndist2 / temp
        # logits = torch.where(inside, logits, torch.full_like(logits, -1e9))
        # weights = torch.softmax(logits, dim=-1)
        weights=1/(1+ (ndist2 / temp)**2)
        weights = torch.where(inside, weights, torch.zeros_like(weights))

        linear_idx = ngx * width + ngy
        linear_idx = torch.where(inside, linear_idx, torch.zeros_like(linear_idx))

        image_flat = torch.zeros((batch_size, height * width), device=device, dtype=dtype)
        image_flat.scatter_add_(
            dim=1,
            index=linear_idx.reshape(batch_size, -1),
            src=weights.reshape(batch_size, -1),
        )

        output = 1.0 - torch.exp(-image_flat.view(batch_size, height, width))
        return output


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
        grid_ids = torch.as_tensor(self.traj_grid_ids[idx], dtype=torch.float32 ).unsqueeze(1)
        coors = self.traj_coor[idx].float()
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