import torch 
import numpy as np 
import pickle as pk
import hashlib
import json
import os
from .grid import Grid


class TrajData(object):
    def __init__(self, dataset, target_measure, dataset_config, grid_size):
        # Load the dataset in a 3:1:6 ratio.
        self.train_start, self.eval_start, self.test_start = 0, 3000, 4000    

        # 数据读取
        traj_data_path = dataset_config[dataset]["traj_data"]
        dis_matrix_path = dataset_config[dataset]["dis_matrix"][target_measure]

        # Cache all derived data because grid conversion and image encoding are
        # repeated for every training run.
        cache_metadata = {
            "version": 1,
            "dataset": dataset,
            "target_measure": target_measure,
            "grid_size": repr(grid_size),
            "dataset_config": repr(dataset_config[dataset]),
            "sources": [
                self._file_signature(traj_data_path),
                self._file_signature(dis_matrix_path),
            ],
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cache", "traj_data"
        )
        cache_path = os.path.join(cache_dir, f"{dataset}_{target_measure}_{cache_key}.pkl")

        if self._load_cache(cache_path, cache_metadata):
            print(f"Loaded TrajData cache: {cache_path}")
            return

        traj_data = pk.load(open(traj_data_path, 'rb'))
        dis_matrix = pk.load(open(dis_matrix_path, 'rb'))

        
        if target_measure == "dtw":
            dis_matrix = dis_matrix / np.max(dis_matrix[:self.eval_start,:self.eval_start]) # follow Neutraj, T3S and TMN, normalize the distance matrix by the maximum value of the training set.
        self.lon_range = dataset_config[dataset]["lon_range"]
        self.lat_range = dataset_config[dataset]["lat_range"]
        data_range = dataset_config[dataset]["data_range"]
        self.mean_lon = data_range["mean_lon"] 
        self.mean_lat = data_range["mean_lat"]
        self.std_lon = data_range["std_lon"]
        self.std_lat = data_range["std_lat"] 

        # 属性统计和数据处理
        self.grid = Grid(delta=grid_size, lon_range=self.lon_range, lat_range=self.lat_range)
        #TODO: 可以测试一下到底是每次都处理快, 还是处理一次存起来快
        # self.traj_data_tensor = []  # 所有轨迹的GPS序列, 列表嵌套tensor的格式
        # traj_lens = []
        # self.x_grid_idx_list, self.y_grid_idx_list, self.grid_idx_list = [], [], []  # grid index, 嵌套列表格式
        # for i in range(len(traj_data)):
        #     temp_traj = torch.tensor(traj_data[i], dtype=torch.float, device="cpu")
        #     x_grid_idxes, y_grid_idxes, grid_idxes = self.grid.traj2grid_seq(traj=temp_traj, list_type="sepa")
        #     self.traj_data_tensor.append(temp_traj)
        #     traj_lens.append(len(temp_traj))
        #     self.x_grid_idx_list.append(x_grid_idxes)
        #     self.y_grid_idx_list.append(y_grid_idxes)
        #     self.grid_idx_list.append(grid_idxes)    
        traj_lens, self.x_grid_idx_list, self.y_grid_idx_list, self.grid_idx_list, self.traj_data_tensor = TrajData.getTrajs(traj_data, self.grid)

        self.traj_lens = torch.tensor(traj_lens, dtype=torch.int32)  # 所有轨迹的长度, tensor格式
        # np.repeat(self.traj_lens[None, :], repeats=len(self.traj_lens), axis=0)  # 这个用到再说
        self.max_traj_len = self.traj_lens.max()
        self.min_traj_len = self.traj_lens.min()
        lon_lat_image_list = self.image_encode(self.x_grid_idx_list, self.y_grid_idx_list, self.grid.grid_num_x, self.grid.grid_num_y)  # 硬编码traj image, 嵌套列表格式
        # lon_lat_image_list_soft = self.image_encode_soft(traj_data, )
        self.lon_lat_image_list = torch.tensor(lon_lat_image_list, dtype=torch.float32)
        
        # 得到对应矩阵
        train_dis_matrix = torch.tensor(dis_matrix[self.train_start:self.eval_start, self.train_start:self.eval_start]).float()
        eval_dis_matrix = torch.tensor(dis_matrix[self.eval_start:self.test_start, self.eval_start:self.test_start]).float()
        test_dis_matrix = torch.tensor(dis_matrix[self.test_start:, self.test_start:]).float()

        # 得到knn距离index矩阵, 用于计算triplet loss
        self.train_knn, self.eval_knn, self.test_knn = np.argsort(train_dis_matrix, axis=1), np.argsort(eval_dis_matrix, axis=1), np.argsort(test_dis_matrix, axis=1)

        # 数据划分: xxx_data = (轨迹, 距离矩阵, 轨迹长度, x轴的grid id, y轴的grid id, 1维的grid id, traj image list, knn距离索引矩阵)
        self.train_data = (self.traj_data_tensor[self.train_start:self.eval_start], train_dis_matrix, self.traj_lens[self.train_start:self.eval_start], 
                           self.x_grid_idx_list[self.train_start:self.eval_start], self.y_grid_idx_list[self.train_start:self.eval_start], 
                           self.grid_idx_list[self.train_start:self.eval_start], self.lon_lat_image_list[self.train_start:self.eval_start], self.train_knn)
        self.eval_data = (self.traj_data_tensor[self.eval_start:self.test_start], eval_dis_matrix, self.traj_lens[self.eval_start:self.test_start], 
                          self.x_grid_idx_list[self.eval_start:self.test_start], self.y_grid_idx_list[self.eval_start:self.test_start], 
                          self.grid_idx_list[self.eval_start:self.test_start], self.lon_lat_image_list[self.eval_start:self.test_start], self.eval_knn)
        self.test_data = (self.traj_data_tensor[self.test_start:], test_dis_matrix, self.traj_lens[self.test_start:], 
                          self.x_grid_idx_list[self.test_start:], self.y_grid_idx_list[self.test_start:], 
                          self.grid_idx_list[self.test_start:], self.lon_lat_image_list[self.test_start:], self.test_knn)
        
        # TODO. Brand New Topic for next paper: Looking for an automated method to determine the value of alpha.
        if target_measure == "fret":
            self.alpha = 8
        if target_measure == "haus":
            self.alpha = 8
        if target_measure == "dtw":
            self.alpha = 16 

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as cache_file:
            pk.dump(
                {"metadata": cache_metadata, "data": self.__dict__},
                cache_file,
                protocol=pk.HIGHEST_PROTOCOL,
            )
        print(f"Saved TrajData cache: {cache_path}")

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
    
    @staticmethod
    def getTrajs(traj_data, grid: Grid):
        traj_lens = []
        x_grid_idx_list, y_grid_idx_list, grid_idx_list = [], [], []
        traj_data_tensor = []

        for i in range(len(traj_data)):
            temp_traj = torch.tensor(traj_data[i], dtype=torch.float, device="cpu")
            x_grid_idxes, y_grid_idxes, grid_idxes = grid.traj2grid_seq(traj=temp_traj, list_type="sepa")
            traj_data_tensor.append(temp_traj)
            traj_lens.append(len(temp_traj))
            x_grid_idx_list.append(x_grid_idxes)
            y_grid_idx_list.append(y_grid_idxes)
            grid_idx_list.append(grid_idxes) 
        return traj_lens, x_grid_idx_list, y_grid_idx_list, grid_idx_list, traj_data_tensor


    @staticmethod
    def _file_signature(path):
        path = os.path.abspath(path)
        stat = os.stat(path)
        return {
            "path": path,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _load_cache(self, cache_path, cache_metadata):
        if not os.path.exists(cache_path):
            return False

        try:
            with open(cache_path, "rb") as cache_file:
                cached = pk.load(cache_file)
            if cached.get("metadata") != cache_metadata:
                return False
            self.__dict__.update(cached["data"])
            return True
        except (OSError, EOFError, KeyError, TypeError, pk.UnpicklingError):
            return False

        
