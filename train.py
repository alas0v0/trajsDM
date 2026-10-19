import time
import torch


def train_one_epoch_convtraj(model, train_dataloader, optimizer, criterion, device, epoch):
        dataloader = train_dataloader.get_dataloader()
        start_time = time.time()
        model.train() 
        epoch_loss = 0
        for idx, batch in enumerate(dataloader):
            inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array, distance_array = batch
                    
            anchor_embedding, positive_embedding, negative_embedding = model(inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array)

            # 加不加再说
            # anchor_embedding = nn.functional.normalize(anchor_embedding, dim=1)
            # positive_embedding = nn.functional.normalize(positive_embedding, dim=1)
            # negative_embedding = nn.functional.normalize(negative_embedding, dim=1)

            positive_distance_target = torch.as_tensor(distance_array[0]).to(device)
            negative_distance_target = torch.as_tensor(distance_array[1]).to(device)
            cross_distance_target = torch.as_tensor(distance_array[2]).to(device)

            positive_learning_distance = torch.linalg.vector_norm(anchor_embedding - positive_embedding, ord=2, dim=1)
            negative_learning_distance = torch.linalg.vector_norm(anchor_embedding - negative_embedding, ord=2, dim=1)
            cross_learning_distance = torch.linalg.vector_norm(positive_embedding - negative_embedding, ord=2, dim=1)

            # forward (triplet loss)
            _, _, loss = criterion(epoch, positive_learning_distance, positive_distance_target, 
                                        negative_learning_distance, negative_distance_target,
                                        cross_learning_distance,    cross_distance_target)
            
            # backward
            epoch_loss += loss.item()  # total loss of this epoch  
            #################### Gradient descent and backpropagation  ####################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.time()  
        return epoch_loss / (idx + 1), end_time - start_time   


def train_one_epoch_trajcl(model, train_dataloader, optimizer, criterion, device):
    dataloader = train_dataloader.get_dataloader()  # 得到torch.utils.data.dataloader.DataLoader类的对象
    similarity = train_dataloader.similarity.to(device)
    # distance = train_dataloader.dis_matrix.to(device)
    start_time = time.time()
    model.train() 
    epoch_loss = 0
    for idx, batch in enumerate(dataloader):

        batch_idx, grid_trajs, padded_trajs_emb_p, traj_lens, src_padding_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
        # batch内打乱traj_ids
        shuffled_batch_idx = torch.randperm(batch_idx.shape[0])
        shuffled_idx = batch_idx[shuffled_batch_idx]

        # 找相似度
        batch_sim = similarity[batch_idx, shuffled_idx]  # 相似度
    
        # forward
        traj_vecs = model(grid_trajs, padded_trajs_emb_p, traj_lens, src_padding_mask)
        loss = criterion(traj_vecs, traj_vecs[shuffled_batch_idx], batch_sim) # mse loss 两个向量之间的距离与真实距离(负对数归一化处理后的结果)计算loss        
                
        # backward
        epoch_loss += loss.item()  # total loss of this epoch  

        #################### Gradient descent and backpropagation  ####################
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)  # Prevent gradient explosion.
        optimizer.step()
    end_time = time.time()  
    return epoch_loss / (idx + 1), end_time - start_time  # 返回每个batch的平均loss, 训练一个epoch所用的时间   

def train_one_epoch_dualsim(model, train_dataloader, optimizer, criterion, device):
    dataloader = train_dataloader.get_dataloader()  # 得到torch.utils.data.dataloader.DataLoader类的对象
    # similarity = train_dataloader.similarity.to(device)
    distance = train_dataloader.dis_matrix.to(device)
    start_time = time.time()
    model.train() 
    epoch_loss = 0
    for idx, batch in enumerate(dataloader):

        batch_idx, trajs, weights, linear_idx, trajs_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)

        # batch内打乱traj_ids
        shuffled_batch_idx = torch.randperm(batch_idx.shape[0])
        shuffled_idx = batch_idx[shuffled_batch_idx]

        # 找相似度
        # batch_sim = similarity[batch_idx, shuffled_idx]  # 相似度
        #distance
        batch_dist = distance[batch_idx, shuffled_idx]
    
        # forward
        traj_vecs = model(trajs, weights, linear_idx, trajs_mask)
        # loss = criterion(traj_vecs, traj_vecs[shuffled_batch_idx], batch_sim) # mse loss 两个向量之间的距离与真实距离(负对数归一化处理后的结果)计算loss        
        loss = criterion(traj_vecs, traj_vecs[shuffled_batch_idx], batch_dist) # mse loss 两个向量之间的距离与真实距离(负对数归一化处理后的结果)计算loss        
                
        # backward
        epoch_loss += loss.item()  # total loss of this epoch  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()  
    return epoch_loss / (idx + 1), end_time - start_time  # 返回每个batch的平均loss, 训练一个epoch所用的时间   

def train_one_epoch_other(model, train_dataloader, optimizer, criterion, device):
    dataloader = train_dataloader.get_dataloader()
    similarity = train_dataloader.similarity.to(device)
    start_time = time.time()
    model.train() 
    epoch_loss = 0
    for idx, batch in enumerate(dataloader):
        padded_trajs, trajs_mask, batch_idx = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        # target trajs idx (padded_trajs打乱)
        shuffled_batch_idx = torch.randperm(padded_trajs.shape[0])
        shuffled_idx = batch_idx[shuffled_batch_idx]

        # similarity
        batch_sim = similarity[batch_idx, shuffled_idx]  # 相似度    

        traj_vecs = model.forward(padded_trajs, trajs_mask)  # 将轨迹编码为向量
        # traj_vecs = nn.functional.normalize(traj_vecs, dim=1)
        loss = criterion(traj_vecs, traj_vecs[shuffled_batch_idx], batch_sim) # mse loss 两个向量之间的距离与真实距离(负对数归一化处理后的结果)计算loss
        # backward
        epoch_loss += loss.item()  # total loss of this epoch  

        #################### Gradient descent and backpropagation  ####################
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)  # Prevent gradient explosion.
        optimizer.step()
    end_time = time.time()  
    return epoch_loss / (idx + 1), end_time - start_time  # 返回每个batch的平均loss, 训练一个epoch所用的时间

