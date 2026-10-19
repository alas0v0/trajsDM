import torch 
import random 
import numpy as np
from torch.utils.data import DataLoader
from utils.data_process import TrajData
from torch.nn.utils.rnn import pad_sequence
from dataset import ConvTrajDatset, TrajCLDataset, TrajSimLeaDataset, TrajDualSimformer
from utils.tools import get_padded_mask, get_normalized_padded_traj


class TrajSimLeaDataLoader():
    def __init__(self, traj_data, args, mode, syn_data = None, grid_mode="soft"):
        # 创建DataSet
        self.mode = mode 
        if self.mode == "train":
            self.split_dataset = traj_data.train_data
        elif self.mode == "eval":
            self.split_dataset = traj_data.eval_data
        elif self.mode == "test":
            self.split_dataset = traj_data.test_data 
        elif self.mode == "syn" and syn_data is not None:
            self.data = syn_data
        else:
            raise ValueError("Not right mode")
        
        self.grid_mode = grid_mode
        self.mean_lon = traj_data.mean_lon 
        self.mean_lat = traj_data.mean_lat 
        self.std_lon = traj_data.std_lon 
        self.std_lat = traj_data.std_lat 
        self.max_traj_len = traj_data.max_traj_len
        self.min_traj_len = traj_data.min_traj_len
        self.lon_range = traj_data.lon_range
        self.lat_range = traj_data.lat_range
        self.grid = traj_data.grid
        self.grid.neigh_num = args.neighbor_num

        self.mean_and_std = (self.mean_lon, self.mean_lat, self.std_lon, self.std_lat)
        self.range = (self.lon_range, self.lat_range)
        #soft_grid
        self.grid.temperature = args.temperature
        print(f"temperature = {self.grid.temperature} ,neigh_num = {self.grid.neigh_num}")
        if self.mode == "syn":
            self.trajs, self.dist_matrix = self.data
            traj_lens, self.x_grid_idx_list, self.y_grid_idx_list, self.grid_idx_list, _ = TrajData.getTrajs(self.trajs, self.grid)
            self.traj_lens = torch.tensor(traj_lens, dtype=torch.int32)  # 所有轨迹的长度, tensor格式
            # 数据划分: xxx_data = (轨迹, 距离矩阵, 轨迹长度, x轴的grid id, y轴的grid id, 1维的grid id)
            self.split_dataset = [self.trajs, self.dist_matrix, self.traj_lens, self.x_grid_idx_list ,self.y_grid_idx_list, self.grid_idx_list]

        if args.model == "ConvTraj":
            self.dataset = ConvTrajDatset(self.split_dataset,self.max_traj_len, self.mean_and_std, self.range, self.grid, self.grid_mode)

        elif args.model == "TrajCL":
            self.dataset = TrajCLDataset(self.split_dataset, self.mean_lon, self.mean_lat, self.std_lon, self.std_lat)
            
        elif args.model == "DualSIMformer":
            self.dataset = TrajDualSimformer(self.split_dataset)

        elif args.model in ["SIMformer", "LSTM", "GRU", "Linear"]:
            self.dataset = TrajSimLeaDataset(self.split_dataset, self.mean_lon, self.mean_lat, self.std_lon, self.std_lat)
        else:
            raise ValueError("Not supported model.")

        # 传递训练参数 
        self.dis_matrix = self.split_dataset[1]
        self.knn = np.argsort(self.dis_matrix.detach().cpu().numpy(), axis=1)
        self.batch_size = args.batch_size
        self.alpha = traj_data.alpha
        self.sampling_num = args.sampling_num
        self.model = args.model 
   

        self.similarity = torch.exp(-self.alpha*self.dis_matrix)  # 预先计算相似度, 避免重复计算

        # 创建DataLoader
        self.dataloader = self.create_dataloader()
    
    def get_dataloader(self):  
        return self.dataloader

    def get_dis_matrix(self):
        return self.dis_matrix 
    
    def collate_fn_convtraj_train(self, data):
        traj_ids, _, _, _ = zip(*data)  # trajs_mask竟然没用到, 出乎意料
        anchor_lon_list,         positive_lon_list,         negative_lon_list       = [], [], []
        anchor_lat_list,         positive_lat_list,         negative_lat_list       = [], [], []
        anchor_image_list,       positive_image_list,       negative_image_list     = [], [], []
        positive_distance_list,  negative_distance_list,    cross_distance_list     = [], [], []

        for anchor_idx in traj_ids:
            positive_sampling_index_list, negative_sampling_index_list = self.triplet_selection(self.knn[anchor_idx], self.dis_matrix[anchor_idx])
            for k in range(self.sampling_num):
                temp_pos_idx, temp_neg_idx = positive_sampling_index_list[k], negative_sampling_index_list[k]

                anchor_lon_list.append(self.dataset.padded_normalized_trajs_coor[anchor_idx, :, 0:1])
                anchor_lat_list.append(self.dataset.padded_normalized_trajs_coor[anchor_idx, :, 1:2])
                anchor_image_list.append(self.dataset.traj_images[anchor_idx])

                positive_lon_list.append(self.dataset.padded_normalized_trajs_coor[temp_pos_idx, :, 0:1])
                positive_lat_list.append(self.dataset.padded_normalized_trajs_coor[temp_pos_idx, :, 1:2])
                positive_image_list.append(self.dataset.traj_images[temp_pos_idx])

                negative_lon_list.append(self.dataset.padded_normalized_trajs_coor[temp_neg_idx, :, 0:1])
                negative_lat_list.append(self.dataset.padded_normalized_trajs_coor[temp_neg_idx, :, 1:2])
                negative_image_list.append(self.dataset.traj_images[temp_neg_idx])

                positive_distance_list.append(self.dis_matrix[anchor_idx][temp_pos_idx])
                negative_distance_list.append(self.dis_matrix[anchor_idx][temp_neg_idx])
                cross_distance_list.append(self.dis_matrix[temp_pos_idx][temp_neg_idx])

        inputs_lon_array = [torch.stack(anchor_lon_list, dim=0), torch.stack(positive_lon_list, dim=0),torch.stack(negative_lon_list, dim=0),]
        inputs_lat_array = [torch.stack(anchor_lat_list, dim=0),torch.stack(positive_lat_list, dim=0),torch.stack(negative_lat_list, dim=0),]
        inputs_lon_lat_image_array = [torch.stack(anchor_image_list, dim=0),torch.stack(positive_image_list, dim=0),torch.stack(negative_image_list, dim=0),]
        distance_array = [torch.stack(positive_distance_list).float(),torch.stack(negative_distance_list).float(),torch.stack(cross_distance_list).float(),]
        return inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array, distance_array

    def collate_fn_trajcl(self, data):
        # traj_ids, traj_coor_dedup, traj_grid_ids_dedup, traj_lens_dedup = zip(*data)  # coors, grid_id原始未去重
        traj_ids, traj_xy, traj_lens = zip(*data)  # coors, grid_id原始未去重
        
        traj_ids = torch.stack(traj_ids, dim=0)  # shape: [batch_size]
        traj_lens = torch.stack(traj_lens).long()
        traj_xy = pad_sequence(list(traj_xy), batch_first=True) #得到padding trajs
        max_len = traj_xy.shape[1]
        trajs_mask = (torch.arange(max_len).unsqueeze(0) < traj_lens.unsqueeze(1))#mask

        weights, linear_idx = self.getWeights(traj_xy[...,1:3], trajs_mask, self.grid)

        # traj_xy, trajs_mask, traj_lens = TrajSimLeaDataLoader._dedup(traj_xy, trajs_mask)               #b, l1, 3
        padded_trajs_emb_p = self._generate_spatial_features(traj_xy[...,1:3], trajs_mask, self.grid)  #embedding
        # padded_traj_grid_ids = traj_xy[...,0].long()
        return traj_ids, (weights, linear_idx), padded_trajs_emb_p, traj_lens, ~trajs_mask
    
    def collate_fn_dualsim(self,data):
        trajs_ids, trajs_xy = zip(*data)
        trajs_ids = torch.stack(trajs_ids, dim=0)

        trajs_xy = pad_sequence(list(trajs_xy), batch_first=True) #得到padding trajs
        trajs_mask = get_padded_mask(trajs_xy)
        if self.grid_mode == "soft":
            weights, linear_idx = self.getWeights(trajs_xy[...,1:3], trajs_mask, self.grid)#B, L, 9
        elif self.grid_mode == "hard":
            linear_idx = trajs_xy[...,0].long().unsqueeze(-1)
            weights = torch.ones(linear_idx.shape, dtype=trajs_xy.dtype, device=trajs_xy.device, )
            
        trajs = get_normalized_padded_traj(trajs_xy[...,1:3], trajs_mask, self.mean_lon, self.mean_lat, self.std_lon, self.std_lat)
        return trajs_ids, trajs, weights, linear_idx, trajs_mask

    def collate_fn_other(self, data):
        trajs, indices = zip(*data)
        padded_trajs = pad_sequence(list(trajs), batch_first=True, padding_value=0)  # shape: [batch_size, max_seq_len, 2]

        trajs_mask = get_padded_mask(padded_trajs)
        normalized_padded_trajs = get_normalized_padded_traj(padded_trajs, trajs_mask, self.mean_lon, self.mean_lat, self.std_lon, self.std_lat )
        return normalized_padded_trajs, trajs_mask, torch.tensor(indices)

    def create_dataloader(self) -> DataLoader:
        pin_memory = False
        if self.mode == "train" or self.mode =="syn":
            if self.model == "ConvTraj":
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, 
                                        pin_memory=pin_memory, collate_fn=self.collate_fn_convtraj_train)
            elif self.model == "TrajCL":
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, 
                                        pin_memory=pin_memory, collate_fn=self.collate_fn_trajcl)  
            elif self.model == "DualSIMformer":
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, 
                        pin_memory=pin_memory, collate_fn=self.collate_fn_dualsim) 
                    
            elif self.model in ["SIMformer", "GRU", "LSTM", "Linear"]:
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, 
                                        pin_memory=pin_memory, collate_fn=self.collate_fn_other) 
            else:
                raise ValueError("不支持该模型")
            
        if self.mode == "test" or self.mode == "eval":
            if self.model == "ConvTraj":
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, 
                                    pin_memory=pin_memory)  # , collate_fn=self.collate_fn_convtraj_test_eval
            elif self.model == "TrajCL":
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, 
                                    pin_memory=pin_memory, collate_fn=self.collate_fn_trajcl)
            elif self.model == "DualSIMformer":
                                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, 
                                                    pin_memory=pin_memory, collate_fn=self.collate_fn_dualsim)  
            elif self.model in ["SIMformer", "GRU", "LSTM", "Linear"]:
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, 
                                pin_memory=pin_memory, collate_fn=self.collate_fn_other)     
            else:
                raise ValueError("不支持该模型")       
                
        return dataloader
    
    # convtraj helper
    def triplet_selection(self, knn, distance_list):
        positive_sampling_index_list, negative_sampling_index_list = [], []

        max_pos = min(200, len(knn) - 1)
        tem_positive_sampling_index_list, tem_negative_sampling_index_list = self.distance_sampling_convtraj(knn, self.sampling_num, distance_list, pos_begin_pos=0, pos_end_pos=max_pos, neg_begin_pos=0, neg_end_pos=max_pos)
        positive_sampling_index_list.extend(tem_positive_sampling_index_list)
        negative_sampling_index_list.extend(tem_negative_sampling_index_list)
        return positive_sampling_index_list, negative_sampling_index_list

    def distance_sampling_convtraj(self, knn, sampling_num, distance_list, pos_begin_pos = 0,
                                                            pos_end_pos = 200,
                                                            neg_begin_pos = 0,
                                                            neg_end_pos = 200):
        positive_sample_index = []
        negative_sample_index = []
        positive_knn_sample   = []
        negative_knn_sample   = []
        for i in range(sampling_num):

            sampling_begin_pos = pos_begin_pos
            sampling_end_pos   = pos_end_pos
            first_num  = random.randint(sampling_begin_pos, sampling_end_pos)

            while (distance_list[knn[first_num]] < 0.0001):
                # print(first_num, distance_list[knn[first_num]])
                first_num  = random.randint(sampling_begin_pos, sampling_end_pos)


            sampling_begin_pos = neg_begin_pos
            sampling_end_pos   = neg_end_pos
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
            while (distance_list[knn[second_num]] < 0.0001):
                second_num  = random.randint(sampling_begin_pos, sampling_end_pos)

            while(second_num == first_num):
                second_num = random.randint(sampling_begin_pos, sampling_end_pos)
            if first_num > second_num:
                first_num, second_num = second_num, first_num

            positive_sample_index.append(knn[first_num])
            negative_sample_index.append(knn[second_num])
            positive_knn_sample.append(first_num)
            negative_knn_sample.append(second_num)
            
        return positive_sample_index, negative_sample_index

    # trajcl helper
    @staticmethod
    def _generate_spatial_features(trajs, mask, grid):
        # trajs: [B, L_max, 2]  float tensor
        # mask: [B, L_max]
        # return [B, L_max, 4] float tensor, (x_norm, y_norm, dist, radian)

        device, dtype = trajs.device , trajs.dtype
        mask = mask.bool()
        #========constant================
        lon_min, lon_max = grid.lon_range
        lat_min, lat_max = grid.lat_range
        delta = grid.delta
        dist_scale = 11.0 * delta / 1.414
  
        #1. normalize x, y
        x = (trajs[..., 0] - lon_min) / (lon_max - lon_min)
        y = (trajs[..., 1] - lat_min) / (lat_max - lat_min)

        #2. 相邻有效点距离
        pair_valid = mask[:, :-1] & mask[:, 1:]  # [B, L-1]
        lens = torch.linalg.vector_norm(trajs[:, 1:, :] - trajs[:, :-1, :], dim=-1)      
        lens = lens * pair_valid.to(dtype)
        #3. 中间点dist : (lens[i-1] + lens[i]) / 2
        dist = (lens[:, :-1] + lens[:, 1:]) / 2
        dist = dist / dist_scale

        middle_mask = mask[:, 1:-1]
        dist = dist * middle_mask.to(dtype)

        p_prev ,p_curr , p_next= trajs[:, :-2], trajs[:, 1:-1], trajs[:, 2:]
        angle_prev = torch.atan2(p_prev[..., 0]-p_curr[..., 0] , p_prev[..., 1]-p_curr[..., 1])
        angle_next = torch.atan2(p_next[..., 0]-p_curr[..., 0] , p_next[..., 1]-p_curr[..., 1])

        radian = torch.pi - angle_prev + angle_next 
        radian = 1 - torch.abs(radian) / torch.pi

        angle_mask = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
        radian = radian * angle_mask.to(dtype)

        middle = torch.stack([x[:, 1:-1], y[:, 1:-1], dist, radian], dim=-1)
        first  = torch.stack([x[:, 0], y[:, 0], torch.zeros_like(x[:, 0]), torch.zeros_like(x[:, 0]) ],dim=-1).unsqueeze(1)
        lengths = mask.sum(dim=1)          # [B]
        last_idx = lengths - 1             # [B]
        batch_idx = torch.arange(trajs.size(0), device=device)
        last_x = x[batch_idx, last_idx]
        last_y = y[batch_idx, last_idx]
        last = torch.stack([last_x, last_y, torch.zeros_like(last_x), torch.zeros_like(last_x)], dim=-1 ).unsqueeze(1)

        # spatial_features = torch.cat([first, middle, last], dim=1)
        spatial_features = torch.cat([first, middle, torch.zeros_like(last)], dim=1)
        spatial_features[batch_idx, last_idx] = last.squeeze(1)
        spatial_features = spatial_features * mask.unsqueeze(-1).to(dtype)

        return spatial_features
    
    @staticmethod
    def _dedup(trajs, mask):
        #(id, x, y) padding + mask
        B, L, C = trajs.shape
        mask = mask.bool()

        grid = trajs[...,0]

        # keep 
        keep = torch.zeros_like(mask)#[B,L]
        keep[:,0] = mask[:,0]

        keep[:,1:] = mask[:,1:] & (grid[:,1:] != grid[:,:-1])#
        #pos
        pos = torch.cumsum(keep.long(), dim=1) -1  
        #len
        lens = keep.sum(dim=1)
        L_new = int(lens.max().item())

        y = torch.full((B, L_new, C), 0, dtype=trajs.dtype, device=trajs.device )
        ymask = torch.zeros(B, L_new, dtype=torch.bool, device=trajs.device)

        batch_idx = torch.arange(B, device=trajs.device)[:, None].expand(B,L)
        y[batch_idx[keep], pos[keep]] = trajs[keep]
        ymask[batch_idx[keep], pos[keep]] = True
        return y, ymask, lens

    @staticmethod
    def getWeights(trajs, valid_mask, grid):
        device = trajs.device
        dtype = trajs.dtype

        grid_size = grid.delta
        lon_min = grid.lon_range[0]
        lat_min = grid.lat_range[0]
        temperature = grid.temperature
        x_num = grid.grid_num_x
        y_num = grid.grid_num_y


        x = (trajs[..., 0]-lon_min) / grid_size
        y = (trajs[..., 1]-lat_min) / grid_size

        center_x = torch.floor(x).to(torch.long)
        center_y = torch.floor(y).to(torch.long)
        valid_bool = valid_mask > 0
        temp = max(float(temperature), 1e-12)

        neighbor_offsets = torch.tensor(
            grid.get_neighbor_offsets(),
            device=device,
            dtype=torch.long,
        )
        offset_x = neighbor_offsets[:, 0].view(1, 1, -1)
        offset_y = neighbor_offsets[:, 1].view(1, 1, -1)

        ngx = center_x.unsqueeze(-1) + offset_x
        ngy = center_y.unsqueeze(-1) + offset_y
        inside = ( valid_bool.unsqueeze(-1) & (ngx >= 0) & (ngx < x_num) & (ngy >= 0) & (ngy < y_num) )

        ndist2 = (x.unsqueeze(-1) - (ngx.to(dtype) + 0.5)) ** 2 + (y.unsqueeze(-1) - (ngy.to(dtype) + 0.5)) ** 2

        logits = -ndist2 / temp
        logits = torch.where(inside, logits, torch.full_like(logits, -1e9))
        weights = torch.softmax(logits, dim=-1)
        # weights=1/(1+ (ndist2 / temp)**2)
        # weights = torch.where(inside, weights, torch.zeros_like(weights))
        # #weights 归一化
        # weight_sum = weights.sum(dim=-1, keepdim=True)
        # weights = weights / weight_sum.clamp_min(1e-12)
        
        linear_idx = ngy * x_num + ngx
        linear_idx = torch.where(inside, linear_idx, torch.zeros_like(linear_idx), )
        return weights, linear_idx
