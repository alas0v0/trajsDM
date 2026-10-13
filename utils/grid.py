import math 
import numpy as np


class Grid(object):
    def __init__(self, delta=0.001, lon_range=[1, 2], lat_range=[1, 2]):

        self.delta = delta 
        self.lon_range = lon_range
        self.lat_range = lat_range 
        self._init_grid_index()

    def _init_grid_index(self):
        x_axis = np.arange(self.lon_range[0], self.lon_range[1], self.delta)
        y_axis = np.arange(self.lat_range[0], self.lat_range[1], self.delta)
        self.grid_num_x = len(x_axis)
        self.grid_num_y = len(y_axis)

    def traj_point2grid_idx(self, p):
        """
        轨迹点映射至网格id, 落在边界上的轨迹点被划分到了下一个格子中, 即左闭右开区间[left, right)
        返回: x轴id, y轴id, 1维索引id
        """
        p_lon, p_lat = p
        x_grid_idx = math.floor((p_lon - self.lon_range[0]) / self.delta)
        y_grid_idx = math.floor((p_lat - self.lat_range[0]) / self.delta)
        grid_idx = y_grid_idx * self.grid_num_x + x_grid_idx 
        return x_grid_idx, y_grid_idx, grid_idx

    def traj2grid_seq(self, traj=[], list_type="sepa"):
        """
        轨迹转化为网格id序列
        list_type为zip: 每个轨迹返回1个list, 该list中每个元素为(x_grid_idx, y_grid_idx, grid_idx);
        list_type为sepa: 每个轨迹返回3个list, 这三个list中每个元素分别对应x_grid_idx, y_grid_idx, grid_idx;
        """
        if list_type == "zip":
            grid_traj = []
            for p in traj:
                x_grid_idx, y_grid_idx, grid_idx = self.traj_point2grid_idx(p)
                grid_traj.append((x_grid_idx, y_grid_idx, grid_idx))
            return grid_traj
        elif list_type == "sepa":
            x_grid_idx_list, y_grid_idx_list, grid_idx_list = [], [], []
            for p in traj:
                x_grid_idx, y_grid_idx, grid_idx = self.traj_point2grid_idx(p)
                x_grid_idx_list.append(x_grid_idx)
                y_grid_idx_list.append(y_grid_idx)
                grid_idx_list.append(grid_idx)
            return x_grid_idx_list, y_grid_idx_list, grid_idx_list
        else:
            raise ValueError(f"list_type只支持sepa和zip")
        
        