import time
import torch 
from utils.top_k import top_k
from utils.data_process import TrajData
from dataset import ConvTrajDatset
from dataloader import TrajSimLeaDataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.tools import pad_traj_tensor_to_fix_length, get_normalized_padded_traj, get_padded_mask, get_normalized_padded_traj_convtraj


def test(model, test_dataloader, args):
    dataloader = test_dataloader.get_dataloader()  # 得到torch.utils.data.dataloader.DataLoader类的对象
    dis_matrix = test_dataloader.get_dis_matrix()  # 得到eval数据集中轨迹两两之间的距离
    mean_lon, mean_lat, std_lon, std_lat = test_dataloader.mean_lon, test_dataloader.mean_lat, test_dataloader.std_lon, test_dataloader.std_lat
    model_name = args.model
    start_time = time.time()
    # 1. Obtain trajectory representation vector
    model.eval()    
    emds = torch.zeros(len(dataloader.dataset), args.hidden_dim).to(args.device) # batch_size * hidden_dim
    with torch.no_grad():
        for batch in dataloader:
            if model_name == "ConvTraj":
                IDs, traj_lons, traj_lats, traj_images = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device), batch[3].to(args.device)
                traj_vecs = model.encode(traj_lons, traj_lats, traj_images) 
            elif model_name == "TrajCL":
                IDs, padded_traj_grid_ids_dedup, padded_trajs_emb_p, traj_lens_dedup, src_padding_mask = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device), batch[3].to(args.device), batch[4].to(args.device)
                traj_vecs = model(padded_traj_grid_ids_dedup, padded_trajs_emb_p, traj_lens_dedup, src_padding_mask)
                # traj_vecs = model(padded_traj_grid_ids_dedup, None, traj_lens_dedup, src_padding_mask)

            elif model_name == "DualSIMformer":
                IDs, trajs, weights, linear_idx, trajs_mask = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device), batch[3].to(args.device), batch[4].to(args.device)
                traj_vecs = model(trajs, weights, linear_idx, trajs_mask)

            elif model_name in ["SIMformer", "GRU", "LSTM", "Linear"]:
                padded_trajs, trajs_mask, IDs = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                traj_vecs = model.forward(padded_trajs, trajs_mask)  # 将轨迹编码为向量
            else:
                raise ValueError("不支持该模型")
            # traj_vecs = nn.functional.normalize(traj_vecs, dim=1)  # 加不加再说
            emds[IDs] = traj_vecs
        # 2. calculate top-k acc
    topk_acc = top_k(emds.cpu(), dis_matrix, metric=args.emb_sim_metric)  
    end_time = time.time() 
    return topk_acc, end_time - start_time

def getEmbeddings(model, model_name, trajs, device, traj_data:TrajData, grid_mode="soft"):
    #trajs list[N]
    model.eval()   
    if model_name == "ConvTraj":
        padded_trajs = pad_traj_tensor_to_fix_length(trajs, traj_data.max_traj_len, pad_value=0.0)
        trajs_mask = get_padded_mask(padded_trajs)
        if grid_mode =="soft":
            images = ConvTrajDatset.image_encode_soft(padded_trajs, trajs_mask, traj_data.grid)
        elif grid_mode == "hard":
            _, x_grid_list,y_grid_list, _, _ = TrajData.getTrajs(trajs, traj_data.grid)
            images = ConvTrajDatset.image_encode(x_grid_list, y_grid_list, traj_data.grid.grid_num_x, traj_data.grid.grid_num_y)
            images = torch.tensor(images,dtype=torch.float32)

        # padded_normalized_trajs =get_normalized_padded_traj_convtraj(padded_trajs, trajs_mask, traj_data.lon_range[0],  traj_data.lat_range[0])
        padded_normalized_trajs = get_normalized_padded_traj(padded_trajs, trajs_mask, traj_data.mean_lon, traj_data.mean_lat, traj_data.std_lon, traj_data.std_lat)
        padded_normalized_trajs = padded_normalized_trajs.to(device)
        images = images.to(device)
        traj_vecs = model.encode(padded_normalized_trajs[:,:, 0:1], padded_normalized_trajs[:,:, 1:2], images) 

    elif model_name == "TrajCL":
        grid_trajs_xy = []
        _, _,_, grid_trajs, _ = TrajData.getTrajs(trajs, traj_data.grid)

        for traj, grid_ids in zip(trajs, grid_trajs):
            traj = traj.float()
            grid_ids = torch.as_tensor(grid_ids, dtype=torch.float32, device=traj.device).unsqueeze(1)  # [L] -> [L, 1]
            grid_traj = torch.cat([grid_ids, traj], dim=1)
            grid_trajs_xy.append(grid_traj)

        traj_lens = torch.tensor([traj.size(0) for traj in trajs], dtype=torch.long)
        padded_trajs = pad_sequence(grid_trajs_xy, batch_first=True) #得到padding trajs
        traj_lens = traj_lens.to(padded_trajs.device)
        max_len = padded_trajs.size(1)
        trajs_mask = (torch.arange(max_len, device=padded_trajs.device).unsqueeze(0) < traj_lens.unsqueeze(1) )
        # padded_trajs, trajs_mask, traj_lens = TrajSimLeaDataLoader._dedup(padded_trajs, trajs_mask)
        padded_trajs_emb_p = TrajSimLeaDataLoader._generate_spatial_features(padded_trajs[...,1:3], trajs_mask, traj_data.grid)#embedding
        padded_trajs = padded_trajs.to(device)
        padded_trajs_emb_p = padded_trajs_emb_p.to(device)
        traj_lens = traj_lens.to(device)
        trajs_mask = trajs_mask.to(device)
        traj_vecs = model(padded_trajs[...,0].long(), padded_trajs_emb_p, traj_lens, ~trajs_mask)

    elif model_name == "DualSIMformer":
        trajs_padding = pad_sequence(trajs, batch_first=True) #得到padding trajs
        trajs_mask = get_padded_mask(trajs_padding)
        if grid_mode == "soft":
            weights, linear_idx = TrajSimLeaDataLoader.getWeights(trajs_padding, trajs_mask, traj_data.grid)
        elif grid_mode == "hard":
            _, _,_, grid_trajs, _ = TrajData.getTrajs(trajs, traj_data.grid)
            grid_trajs = [torch.tensor(grid_ids, dtype=torch.long) for grid_ids in grid_trajs]
            padded_grid_trajs = pad_sequence(grid_trajs, batch_first=True, padding_value=0)
            linear_idx = padded_grid_trajs.long().unsqueeze(-1)
            weights = torch.ones(linear_idx.shape, dtype=trajs_padding.dtype, device=trajs_padding.device)
        
        padded_normalized_trajs = get_normalized_padded_traj(trajs_padding, trajs_mask, traj_data.mean_lon, traj_data.mean_lat, traj_data.std_lon, traj_data.std_lat)
        padded_normalized_trajs,  weights, linear_idx, trajs_mask = padded_normalized_trajs.to(device),  weights.to(device), linear_idx.to(device), trajs_mask.to(device)
        traj_vecs = model.forward(padded_normalized_trajs, weights, linear_idx, trajs_mask)

    elif model_name in ["SIMformer", "GRU", "LSTM", "Linear"]:
        padded_trajs = pad_sequence(list(trajs), batch_first=True) #得到padding trajs
        trajs_mask = get_padded_mask(padded_trajs)
        padded_trajs = get_normalized_padded_traj(padded_trajs, trajs_mask,traj_data.mean_lon,  traj_data.mean_lat,  traj_data.std_lon,  traj_data.std_lat) 
        padded_trajs, trajs_mask = padded_trajs.to(device), trajs_mask.to(device)
        traj_vecs = model.forward(padded_trajs, trajs_mask)  # 将轨迹编码为向量
    return traj_vecs

def getEmbeddingsFromFull(model, model_name, trajs, args, traj_data:TrajData, batch_size=128, required_grad=False):
    embedding_batches = []
    model.eval()
    grad_context = torch.enable_grad() if required_grad else torch.no_grad()
    with grad_context:
        for start in range(0, len(trajs), batch_size):
            batch_trajs = trajs[start:start + batch_size]
            batch_embeddings = getEmbeddings(model, model_name, batch_trajs, args.device, traj_data, args.grid_mode)
            embedding_batches.append(batch_embeddings)
    return torch.cat(embedding_batches, dim=0)