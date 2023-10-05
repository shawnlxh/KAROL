import argparse
import logging
from model import *
from loss import * 
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.loader import HGTLoader
from data import *

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_file", default='./raw_data_food.csv', type=str,
                        help="Data path for training.")
    parser.add_argument("--data_name", default='clothes', type=str,
                        help="Data path for training.")
    parser.add_argument("--contrast_type", type=str, default='dropout', choices=['dropout', 'noise', 'all'], 
                        help="Contrast type")
    parser.add_argument("--output_dir", default='./checkpoints', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--generation_output_dir", default='./generation_output', type=str,
                        help="The output directory where the log will be written.")
    
    parser.add_argument("--epochs", default=50, type=int, help="total epochs")
    parser.add_argument("--train_batch_size", default=256, type=int,help="Batch size for training.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--hidden_channels", default=64, type=int,
                        help="Embedding Size fo model")
    parser.add_argument("--cl_regularization", default=0.01, type=float, 
                        help="regularization coefficient for contrast learning")
    args = parser.parse_args()
    return args

def prepare(args):
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("Training/evaluation parameters %s", args)
    return 
    
def prepare_model(args):
    multi_task_model = Multi_task_model(args.hidden_channels)
    ssl_block = SSL_block(args.hidden_channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    multi_task_model = multi_task_model.to(device)
    ssl_block = ssl_block.to(device)
    optimizer = torch.optim.Adam(multi_task_model.parameters(), lr=args.learning_rate)
    optimizer_ssl = torch.optim.Adam(multi_task_model.parameters(), lr=args.learning_rate ,weight_decay=args.weight_decay)
    return multi_task_model, ssl_block, optimizer, optimizer_ssl,device

##prepare for transe
def sample_neg(train_g,all_g,r,epoch):
    train_e = train_g[r].edge_label_index[0][train_g[r].edge_label==1]
    neg_t = []
    neg_label = torch.tensor([0])
    for e in train_e:
        temp_neg = []
        count = 0
        pos_t = all_g[r].edge_index[1][all_g[r].edge_index[0]==e]
        while True:
            rand_t = torch.randint(all_g[r[2]].node_id.size()[0],(1,))
            if rand_t not in pos_t:
                temp_neg.append(rand_t)
                count += 1
            if count == epoch:
                break
        neg_t.append(temp_neg)
    neg_t = torch.tensor(neg_t)
    edge_label_index_h = train_e
    edge_label_index_t = neg_t
    #edge_label_index = torch.stack((train_e,neg_t),dim = 0)
    edge_label = neg_label.expand_as(train_e)
    return edge_label_index_h,edge_label_index_t, edge_label



def data_loader_i2s(train_data,data,epochs):
    #dataloader for i2s transe
    num_sample_neighbors = [15,15]

    edge_label_index_i2s_pos = train_data['item', 'i2s', 'seller'].edge_index
    neg_label = torch.tensor([0])
    pos_label = torch.tensor([1])
    edge_label_i2s_pos = pos_label.repeat(edge_label_index_i2s_pos.size()[1])
    train_data[("item", "i2s", "seller")].edge_label_index = edge_label_index_i2s_pos
    train_data[("item", "i2s", "seller")].edge_label = edge_label_i2s_pos
    edge_label_index_i2s_neg_h, edge_label_index_i2s_neg_t, edge_label_i2s_neg = sample_neg\
        (train_data,data,("item", "i2s", "seller"),epochs)
    return edge_label_index_i2s_pos, edge_label_i2s_pos, edge_label_index_i2s_neg_h, 
           edge_label_index_i2s_neg_t, edge_label_i2s_neg
    

def data_loader(train_data,data,epochs):
    num_sample_neighbors = [15,15]
    
    # Define seed edges:
    pos_i = train_data["user", "u2i", "item"].edge_label==1
    edge_label_index_u2i_pos = train_data["user", "u2i", "item"].edge_label_index[:,pos_i]
    edge_label_u2i_pos = train_data["user", "u2i", "item"].edge_label[pos_i]
    train_loader_u2i = LinkNeighborLoader(
        data=train_data.edge_type_subgraph([('user', 'u2i', 'item'),('item', 'i2u', 'user')]),
        num_neighbors=num_sample_neighbors, 
        neg_sampling_ratio=0, 
        edge_label_index=(("user", "u2i", "item"), edge_label_index_u2i_pos),
        edge_label=edge_label_u2i_pos,
        batch_size=256,
        shuffle=False,
    )
    pos_i = train_data["user", "u2s", "seller"].edge_label==1
    edge_label_index_u2s_pos = train_data["user", "u2s", "seller"].edge_label_index[:,pos_i]
    edge_label_u2s_pos = train_data["user", "u2s", "seller"].edge_label[pos_i]
    train_loader_u2s = LinkNeighborLoader(
        data=train_data.edge_type_subgraph([('user', 'u2s', 'seller'),('seller', 's2u', 'user')]),
        num_neighbors=num_sample_neighbors, 
        neg_sampling_ratio=0, 
        edge_label_index=(("user", "u2s", "seller"), edge_label_index_u2s_pos),
        edge_label=edge_label_u2s_pos,
        batch_size=256,
        shuffle=False,
    )

    #Define negative graph
    edge_label_index_u2i_neg_h,edge_label_index_u2i_neg_t, edge_label_u2i_neg = 
    sample_neg(train_data,data,('user', 'u2i', 'item'),epochs)
    edge_label_index_u2s_neg_h,edge_label_index_u2s_neg_t, edge_label_u2s_neg = 
    sample_neg(train_data,data,("user", "u2s", "seller"),epochs)
    return train_loader_u2i,train_loader_u2s,edge_label_index_u2i_neg_h,edge_label_index_u2i_neg_t, edge_label_u2i_neg,
           edge_label_index_u2s_neg_h,edge_label_index_u2s_neg_t, edge_label_u2s_neg

def train(args,multi_task_model,train_data,data,ssl_block, optimizer, optimizer_ssl,device):
    edge_label_index_i2s_pos, edge_label_i2s_pos, edge_label_index_i2s_neg_h, 
           edge_label_index_i2s_neg_t, edge_label_i2s_neg = data_loader_i2s(train_data,data,args.epochs)
    train_loader_u2i,train_loader_u2s,edge_label_index_u2i_neg_h,edge_label_index_u2i_neg_t, edge_label_u2i_neg,
           edge_label_index_u2s_neg_h,edge_label_index_u2s_neg_t, edge_label_u2s_neg = data_loader(train_data,data,epochs)
    #ssl loss
    num_sample_neighbors = [30,30]
    perturbed = False
    cl_type = 'infonce' # infonce or max margin
    for epoch in range(args.epochs):
        #frist train transe
        total_loss = 0
        print('begin transe') 
        transe_loss = 0
        train_loader_i2s = LinkNeighborLoader(
            data=train_data.edge_type_subgraph([('item', 'i2s', 'seller'),('seller', 's2i', 'item')]),
            num_neighbors=num_sample_neighbors, 
            neg_sampling_ratio=0, 
            edge_label_index=(("item", "i2s", "seller"), edge_label_index_i2s_pos),
            edge_label=edge_label_i2s_pos,
            batch_size=args.train_batch_size,
            shuffle=False,
        )
        edge_label_index_neg = torch.stack((edge_label_index_i2s_neg_h,edge_label_index_i2s_neg_t[:,epoch]),dim = 0)
        train_loader_i2s_neg = LinkNeighborLoader(
        data=train_data.edge_type_subgraph([('item', 'i2s', 'seller'),('seller', 's2i', 'item')]),
        num_neighbors=num_sample_neighbors, 
        neg_sampling_ratio=0, 
        edge_label_index=(('item', 'i2s', 'seller'), edge_label_index_neg),
        edge_label=edge_label_i2s_neg,
        batch_size=args.train_batch_size,
        shuffle=False)
        for (sampled_data, sampled_data_neg) in \
        tqdm(iter(zip(train_loader_i2s,train_loader_i2s_neg)),total=len(train_loader_i2s)):
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device, non_blocking=True)
            sampled_data_neg = sampled_data_neg.to(device, non_blocking=True)
            x_dict, rel_emb = multi_task_model(sampled_data,transe = True)
            edge_attr = torch.tensor([2]).to(device, non_blocking=True)
            r_emb = rel_emb(edge_attr) #relation for i2s
            x_dict_neg, _ = multi_task_model(sampled_data_neg, transe = True)
            #print(pred[:batch_size].size())
            #print(sampled_data["user", "u2i", "item"].edge_label[:batch_size].size())
            #print(sampled_data["item", "i2s", "seller"].edge_label_index[0])
            h_index = sampled_data["item", "i2s", "seller"].edge_label_index[0]
            #print(h_index)
            h_emb = x_dict['item'][h_index]
            #print(h_emb.size())
            t_pos_index = sampled_data["item", "i2s", "seller"].edge_label_index[1]
            t_neg_index = sampled_data_neg["item", "i2s", "seller"].edge_label_index[1]
            t_pos_emb = x_dict['seller'][t_pos_index]
            t_neg_emb = x_dict_neg['seller'][t_neg_index]
            #print(t_neg_emb.size())
            pos_score = torch.sum(torch.pow(h_emb + r_emb - t_pos_emb, 2), dim=1)
            neg_score = torch.sum(torch.pow(h_emb + r_emb - t_neg_emb, 2), dim=1)
            kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
            kg_loss = torch.mean(kg_loss)
            kg_loss.backward()
            optimizer.step()
            total_loss += float(kg_loss) 
            transe_loss += float(kg_loss)
            #total_examples += pred.numel()
        print('transe finished')

        #begin bpr for u2i
        bpr_u2i_loss = 0
        edge_label_index_neg = torch.stack((edge_label_index_u2i_neg_h,edge_label_index_u2i_neg_t[:,epoch]),dim = 0)
        train_loader_u2i_neg = LinkNeighborLoader(
        data=train_data.edge_type_subgraph([('user', 'u2i', 'item'),('item', 'i2u', 'user')]),
        num_neighbors=num_sample_neighbors, 
        neg_sampling_ratio=0, 
        edge_label_index=(("user", "u2i", "item"), edge_label_index_neg),
        edge_label=edge_label_u2i_neg,
        batch_size=args.train_batch_size,
        shuffle=False)
        for (sampled_data, sampled_data_neg) in \
        tqdm(iter(zip(train_loader_u2i,train_loader_u2i_neg)),total=len(train_loader_u2i)):
            sampled_data = sampled_data.to('cpu')
            sampled_data_neg = sampled_data_neg.to('cpu')
            train_loader_i2s = NeighborLoader(
                train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                num_neighbors=[10,30],
                batch_size=sampled_data['item'].node_id.size()[0],
                input_nodes=('item', sampled_data['item'].node_id))
            train_loader_i2s_neg = NeighborLoader(
                train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                num_neighbors=[10,30],
                batch_size=sampled_data_neg['item'].node_id.size()[0],
                input_nodes=('item', sampled_data_neg['item'].node_id))
            for i, (sampled_i2s, sampled_i2s_neg) in enumerate(zip(train_loader_i2s,train_loader_i2s_neg)):
                batch_len = sampled_i2s['item'].input_id.size()[0]
                batch_len_neg = sampled_i2s_neg['item'].input_id.size()[0]
                edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                sampled_i2s = sampled_i2s.to(device, non_blocking=True)
                sampled_i2s_neg = sampled_i2s_neg.to(device, non_blocking=True)
                x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1',edge_attr = edge_attr)
                x_dict2 = multi_task_model(sampled_i2s_neg, stage = 'stage1',edge_attr = edge_attr)
            sampled_data = sampled_data.to(device, non_blocking=True)
            sampled_data_neg = sampled_data_neg.to(device, non_blocking=True)
            sampled_data['item'].x = x_dict1['item'][:batch_len]
            sampled_data_neg['item'].x = x_dict2['item'][:batch_len_neg]
            pred = multi_task_model(sampled_data)
            pred_neg = multi_task_model(sampled_data_neg)
            loss = bpr_loss(pred,pred_neg)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) 
            bpr_u2i_loss += float(loss) 
        print('bpr for u2i finished')

        #begin bpr for u2s
        bpr_u2s_loss = 0
        edge_label_index_neg = torch.stack((edge_label_index_u2s_neg_h,edge_label_index_u2s_neg_t[:,epoch]),dim = 0)
        train_loader_u2s_neg = LinkNeighborLoader(
        data=train_data.edge_type_subgraph([('user', 'u2s', 'seller'),('seller', 's2u', 'user')]),
        num_neighbors=[10,10], 
        neg_sampling_ratio=0, 
        edge_label_index=(("user", "u2s", "seller"), edge_label_index_neg),
        edge_label=edge_label_u2s_neg,
        batch_size=args.train_batch_size,
        shuffle=False)
        for (sampled_data, sampled_data_neg) in \
        tqdm(iter(zip(train_loader_u2s,train_loader_u2s_neg)),total=len(train_loader_u2s)):
            sampled_data = sampled_data.to('cpu')
            sampled_data_neg = sampled_data_neg.to('cpu')
            train_loader_i2s = NeighborLoader(
                train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                num_neighbors=[30,10],
                batch_size=sampled_data['seller'].node_id.size()[0],
                input_nodes=('seller', sampled_data['seller'].node_id))
            train_loader_i2s_neg = NeighborLoader(
                train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                num_neighbors=[30,10],
                batch_size=sampled_data_neg['seller'].node_id.size()[0],
                input_nodes=('seller', sampled_data_neg['seller'].node_id))
            for i, (sampled_i2s, sampled_i2s_neg) in enumerate(zip(train_loader_i2s,train_loader_i2s_neg)):
                batch_len = sampled_i2s['seller'].input_id.size()[0]
                batch_len_neg = sampled_i2s_neg['seller'].input_id.size()[0]
                edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                sampled_i2s = sampled_i2s.to(device, non_blocking=True)
                sampled_i2s_neg = sampled_i2s_neg.to(device, non_blocking=True)
                x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1',edge_attr = edge_attr)
                x_dict2 = multi_task_model(sampled_i2s_neg, stage = 'stage1',edge_attr = edge_attr)
            sampled_data = sampled_data.to(device, non_blocking=True)
            sampled_data_neg = sampled_data_neg.to(device, non_blocking=True)
            sampled_data['seller'].x = x_dict1['seller'][:batch_len]
            sampled_data_neg['seller'].x = x_dict2['seller'][:batch_len_neg]
            pred = multi_task_model(sampled_data)
            pred_neg = multi_task_model(sampled_data_neg)
            loss = bpr_loss(pred,pred_neg)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) 
            bpr_u2s_loss += float(loss) 
        print('bpr for u2s finished')

        if args.contrast_type == 'noise' or args.contrast_type == 'all':
            print('begin ssl')
            ssl_i_loss = 0
            u2i_u = torch.randperm(train_data['user'].node_id.size()[0])
            train_loader_u2i_ssl = NeighborLoader(
            data=train_data.edge_type_subgraph([('user', 'u2i', 'item'),('item', 'i2u', 'user')]),
            num_neighbors=[10,10], 
            batch_size=args.train_batch_size,
            input_nodes = ('user',u2i_u),
            shuffle=False
            )
            batch = 0
            ssl_temp_i_avg = 0
            for sampled_data1 in \
            tqdm(iter(train_loader_u2i_ssl),total=len(train_loader_u2i_ssl)):
                sampled_data1 = sampled_data1.to('cpu')
                train_loader_i2s = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[5,30],
                    batch_size=sampled_data1['item'].node_id.size()[0],
                    input_nodes=('item', sampled_data1['item'].node_id))
                for i, sampled_i2s in enumerate(train_loader_i2s):
                    batch_len = sampled_i2s['item'].input_id.size()[0]
                    edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                    sampled_i2s = sampled_i2s.to(device, non_blocking=True)
                    x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1', edge_attr = edge_attr)
                sampled_data1 = sampled_data1.to(device, non_blocking=True)
                sampled_data1['item'].x = x_dict1['item'][:batch_len]
                train_len = sampled_data1['user'].input_id.size()[0]
                batch_index = torch.arange(train_len)
                batch_index = batch_index.to(device, non_blocking=True)
                x_dict1 = multi_task_model(sampled_data1 ,stage = 'stage2',device = device, perturbed = True)
                x_dict2 = multi_task_model(sampled_data1 ,stage = 'stage2',device = device, perturbed = True)
                if cl_type == 'infonce':
                    pos,tot,ssl_temp = ssl_block(x_dict1['user'], x_dict2['user'],batch_index,batch_index)
                    #print('res',pos/tot)
                    batch += 1
                    ssl_temp_i_avg += ssl_temp
                    infonce_loss = -torch.mean(torch.log(pos/tot))
                    infonce_loss = ssl_reg * infonce_loss
                    infonce_loss.backward()
                    optimizer_ssl.step()
                    total_loss += float(infonce_loss) 
                    ssl_i_loss += float(infonce_loss) 
            ssl_temp_i_avg = ssl_temp_i_avg/batch
            print('ssl_temp_i_avg', ssl_temp_i_avg)
            print('ssl for item finished')

            #begin ssl between item and seller 
            ssl_is_loss = 0
            train_loader_u2s_ssl = NeighborLoader(
            data=train_data.edge_type_subgraph([('user', 'u2s', 'seller'),('seller', 's2u', 'user')]),
            num_neighbors=[5,5], 
            batch_size=args.train_batch_size,
            input_nodes = ('user',u2i_u),
            shuffle=False
            )
            batch = 0
            ssl_temp_is_avg = 0
            for (sampled_data1, sampled_data2) in \
            tqdm(iter(zip(train_loader_u2i_ssl,train_loader_u2s_ssl)),total=len(train_loader_u2i_ssl)):
                sampled_data1 = sampled_data1.to('cpu')
                sampled_data2 = sampled_data2.to('cpu')
                train_loader_i2s = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[10,30],
                    batch_size=sampled_data1['item'].node_id.size()[0],
                    input_nodes=('item', sampled_data1['item'].node_id))
                train_loader_i2s_neg = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[30,10],
                    batch_size=sampled_data2['seller'].node_id.size()[0],
                    input_nodes=('seller', sampled_data2['seller'].node_id))
                for i, (sampled_i2s, sampled_i2s_neg) in enumerate(zip(train_loader_i2s,train_loader_i2s_neg)):
                    batch_len = sampled_i2s['item'].input_id.size()[0]
                    batch_len_neg = sampled_i2s_neg['seller'].input_id.size()[0]
                    edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                    sampled_i2s = sampled_i2s.to(device, non_blocking=True)
                    sampled_i2s_neg = sampled_i2s_neg.to(device, non_blocking=True)
                    x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1',edge_attr = edge_attr)
                    x_dict2 = multi_task_model(sampled_i2s_neg, stage = 'stage1', edge_attr = edge_attr)
                sampled_data1 = sampled_data1.to(device, non_blocking=True)
                sampled_data2 = sampled_data2.to(device, non_blocking=True)
                sampled_data1['item'].x = x_dict1['item'][:batch_len]
                sampled_data2['seller'].x = x_dict2['seller'][:batch_len_neg]
                train_len = sampled_data1['user'].input_id.size()[0]
                batch_index = torch.arange(train_len)
                batch_index = batch_index.to(device, non_blocking=True)
                x_dict1 = multi_task_model(sampled_data1 ,stage = 'stage2',device = device, perturbed = True)
                #print('x_dict1',x_dict1['user'])
                x_dict2 = multi_task_model(sampled_data2, stage = 'stage2',device = device, perturbed = True)
                if cl_type == 'infonce':
                    pos,tot,ssl_temp = ssl_block(x_dict1['user'], x_dict2['user'],batch_index,batch_index)
                    #print('res',pos/tot)
                    batch += 1
                    ssl_temp_is_avg += ssl_temp
                    infonce_loss = -torch.sum(torch.log(pos/tot))
                    infonce_loss = ssl_reg * infonce_loss
                    infonce_loss.backward()
                    optimizer_ssl.step()
                    total_loss += float(infonce_loss) 
                    ssl_is_loss += float(infonce_loss) 
            ssl_temp_is_avg = ssl_temp_is_avg/batch
            print('ssl_temp_is_avg',ssl_temp_is_avg)
            print('ssl for item seller finished')

            batch = 0
            ssl_temp_i_avg = 0
            for sampled_data1 in \
            tqdm(iter(train_loader_u2s_ssl),total=len(train_loader_u2s_ssl)):
                sampled_data1 = sampled_data1.to('cpu')
                train_loader_i2s = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[30,10],
                    batch_size=sampled_data1['seller'].node_id.size()[0],
                    input_nodes=('seller', sampled_data1['seller'].node_id))
                for i, sampled_i2s in enumerate(train_loader_i2s):
                    batch_len = sampled_i2s['seller'].input_id.size()[0]
                    edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                    sampled_i2s = sampled_i2s.to(device, non_blocking=True)
                    sampled_i2s_neg = sampled_i2s_neg.to(device, non_blocking=True)
                    x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1', edge_attr = edge_attr)
                sampled_data1 = sampled_data1.to(device, non_blocking=True)
                sampled_data1['seller'].x = x_dict1['seller'][:batch_len]
                train_len = sampled_data1['user'].input_id.size()[0]
                batch_index = torch.arange(train_len)
                batch_index = batch_index.to(device, non_blocking=True)
                x_dict1 = multi_task_model(sampled_data1 ,stage = 'stage2', device = device, perturbed = True)
                #print('x_dict1',x_dict1['user'])
                x_dict2 = multi_task_model(sampled_data1, stage = 'stage2', device = device, perturbed = True)
                if cl_type == 'infonce':
                    pos,tot,ssl_temp = ssl_block(x_dict1['user'], x_dict2['user'],batch_index,batch_index)
                    #print('res',pos/tot)
                    batch += 1
                    ssl_temp_s_avg += ssl_temp
                    infonce_loss = -torch.sum(torch.log(pos/tot))
                    infonce_loss = ssl_reg * infonce_loss
                    infonce_loss.backward()
                    optimizer_ssl.step()
                    total_loss += float(infonce_loss) 
                    ssl_s_loss += float(infonce_loss) 
            ssl_temp_s_avg = ssl_temp_s_avg/batch
            print('ssl_temp_s_avg',ssl_temp_s_avg)    
            print('ssl for item seller finished')

        if args.contrast_type == 'dropout' or args.contrast_type == 'all':
            #begin ssl  
            print('begin ssl')
            ssl_i_loss = 0
            u2i_u = torch.randperm(train_data['user'].node_id.size()[0])
            train_loader_u2i_ssl = NeighborLoader(
            data=train_data.edge_type_subgraph([('user', 'u2i', 'item'),('item', 'i2u', 'user')]),
            num_neighbors=[10,10], 
            batch_size=args.train_batch_size,
            input_nodes = ('user',u2i_u),
            shuffle=False
            )
            train_loader_u2i_ssl2 = NeighborLoader(
            data=train_data.edge_type_subgraph([('user', 'u2i', 'item'),('item', 'i2u', 'user')]),
            num_neighbors=[10,10], 
            batch_size=256,
            input_nodes = ('user',u2i_u),
            shuffle=False
            )
            batch = 0
            ssl_temp_i_avg = 0
            for (sampled_data1, sampled_data2) in \
            tqdm(iter(zip(train_loader_u2i_ssl,train_loader_u2i_ssl2)),total=len(train_loader_u2i_ssl)):
                sampled_data1 = sampled_data1.to('cpu')
                sampled_data2 = sampled_data2.to('cpu')
                train_loader_i2s = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[5,30],
                    batch_size=sampled_data1['item'].node_id.size()[0],
                    input_nodes=('item', sampled_data1['item'].node_id))
                train_loader_i2s_neg = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[5,30],
                    batch_size=sampled_data2['item'].node_id.size()[0],
                    input_nodes=('item', sampled_data2['item'].node_id))
                for i, (sampled_i2s, sampled_i2s_neg) in enumerate(zip(train_loader_i2s,train_loader_i2s_neg)):
                    batch_len = sampled_i2s['item'].input_id.size()[0]
                    batch_len_neg = sampled_i2s_neg['item'].input_id.size()[0]
                    edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                    sampled_i2s = sampled_i2s.to(device, non_blocking=True)
                    sampled_i2s_neg = sampled_i2s_neg.to(device, non_blocking=True)
                    x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1', edge_attr = edge_attr)
                    x_dict2 = multi_task_model(sampled_i2s_neg, stage = 'stage1', edge_attr = edge_attr)
                sampled_data1 = sampled_data1.to(device, non_blocking=True)
                sampled_data2 = sampled_data2.to(device, non_blocking=True)
                sampled_data1['item'].x = x_dict1['item'][:batch_len]
                sampled_data2['item'].x = x_dict2['item'][:batch_len_neg]
                train_len = sampled_data1['user'].input_id.size()[0]
                batch_index = torch.arange(train_len)
                batch_index = batch_index.to(device, non_blocking=True)
                x_dict1 = multi_task_model(sampled_data1 ,stage = 'stage2')
                #print('x_dict1',x_dict1['user'])
                x_dict2 = multi_task_model(sampled_data2, stage = 'stage2')
                pos,tot,ssl_temp = ssl_block(x_dict1['user'], x_dict2['user'],batch_index,batch_index)
                #print('res',pos/tot)
                batch += 1
                ssl_temp_i_avg += ssl_temp
                infonce_loss = -torch.mean(torch.log(pos/tot))
                infonce_loss = ssl_reg * infonce_loss
                infonce_loss.backward()
                optimizer_ssl.step()
                total_loss += float(infonce_loss) 
                ssl_i_loss += float(infonce_loss) 
            ssl_temp_i_avg = ssl_temp_i_avg/batch
            print('ssl_temp_i_avg', ssl_temp_i_avg)
            print('ssl for item finished')

            #begin ssl between item and seller 
            ssl_is_loss = 0
            train_loader_u2s_ssl = NeighborLoader(
            data=train_data.edge_type_subgraph([('user', 'u2s', 'seller'),('seller', 's2u', 'user')]),
            num_neighbors=[5,5], 
            batch_size=args.train_batch_size,
            input_nodes = ('user',u2i_u),
            shuffle=False
            )
            batch = 0
            ssl_temp_is_avg = 0
            for (sampled_data1, sampled_data2) in \
            tqdm(iter(zip(train_loader_u2i_ssl,train_loader_u2s_ssl)),total=len(train_loader_u2i_ssl)):
                sampled_data1 = sampled_data1.to('cpu')
                sampled_data2 = sampled_data2.to('cpu')
                train_loader_i2s = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[10,30],
                    batch_size=sampled_data1['item'].node_id.size()[0],
                    input_nodes=('item', sampled_data1['item'].node_id))
                train_loader_i2s_neg = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[30,10],
                    batch_size=sampled_data2['seller'].node_id.size()[0],
                    input_nodes=('seller', sampled_data2['seller'].node_id))
                for i, (sampled_i2s, sampled_i2s_neg) in enumerate(zip(train_loader_i2s,train_loader_i2s_neg)):
                    batch_len = sampled_i2s['item'].input_id.size()[0]
                    batch_len_neg = sampled_i2s_neg['seller'].input_id.size()[0]
                    edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                    sampled_i2s = sampled_i2s.to(device, non_blocking=True)
                    sampled_i2s_neg = sampled_i2s_neg.to(device, non_blocking=True)
                    x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1',edge_attr = edge_attr)
                    x_dict2 = multi_task_model(sampled_i2s_neg, stage = 'stage1', edge_attr = edge_attr)
                sampled_data1 = sampled_data1.to(device, non_blocking=True)
                sampled_data2 = sampled_data2.to(device, non_blocking=True)
                sampled_data1['item'].x = x_dict1['item'][:batch_len]
                sampled_data2['seller'].x = x_dict2['seller'][:batch_len_neg]
                train_len = sampled_data1['user'].input_id.size()[0]
                batch_index = torch.arange(train_len)
                batch_index = batch_index.to(device, non_blocking=True)
                x_dict1 = multi_task_model(sampled_data1 ,stage = 'stage2')
                #print('x_dict1',x_dict1['user'])
                x_dict2 = multi_task_model(sampled_data2, stage = 'stage2')
                pos,tot,ssl_temp = ssl_block(x_dict1['user'], x_dict2['user'],batch_index,batch_index)
                #print('res',pos/tot)
                batch += 1
                ssl_temp_is_avg += ssl_temp
                infonce_loss = -torch.sum(torch.log(pos/tot))
                infonce_loss = ssl_reg * infonce_loss
                infonce_loss.backward()
                optimizer_ssl.step()
                total_loss += float(infonce_loss) 
                ssl_is_loss += float(infonce_loss) 
            ssl_temp_is_avg = ssl_temp_is_avg/batch
            print('ssl_temp_is_avg',ssl_temp_is_avg)
            print('ssl for item seller finished')

            #begin ssl for seller
            ssl_s_loss = 0
            train_loader_u2s_ssl2 = NeighborLoader(
            data=train_data.edge_type_subgraph([('user', 'u2s', 'seller'),('seller', 's2u', 'user')]),
            num_neighbors=[5,5], 
            batch_size=args.train_batch_size,
            input_nodes = ('user',u2i_u),
            shuffle=False
            ) 
            batch = 0
            ssl_temp_s_avg = 0
            for (sampled_data1, sampled_data2) in \
            tqdm(iter(zip(train_loader_u2s_ssl,train_loader_u2s_ssl2)),total=len(train_loader_u2s_ssl)):
                sampled_data1 = sampled_data1.to('cpu')
                sampled_data2 = sampled_data2.to('cpu')
                train_loader_i2s = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[30,10],
                    batch_size=sampled_data1['seller'].node_id.size()[0],
                    input_nodes=('seller', sampled_data1['seller'].node_id))
                train_loader_i2s_neg = NeighborLoader(
                    train_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[30,10],
                    batch_size=sampled_data2['seller'].node_id.size()[0],
                    input_nodes=('seller', sampled_data2['seller'].node_id))
                for i, (sampled_i2s, sampled_i2s_neg) in enumerate(zip(train_loader_i2s,train_loader_i2s_neg)):
                    batch_len = sampled_i2s['seller'].input_id.size()[0]
                    batch_len_neg = sampled_i2s_neg['seller'].input_id.size()[0]
                    edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                    sampled_i2s = sampled_i2s.to(device, non_blocking=True)
                    sampled_i2s_neg = sampled_i2s_neg.to(device, non_blocking=True)
                    x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1', edge_attr = edge_attr)
                    x_dict2 = multi_task_model(sampled_i2s_neg, stage = 'stage1', edge_attr = edge_attr)
                sampled_data1 = sampled_data1.to(device, non_blocking=True)
                sampled_data2 = sampled_data2.to(device, non_blocking=True)
                sampled_data1['seller'].x = x_dict1['seller'][:batch_len]
                sampled_data2['seller'].x = x_dict2['seller'][:batch_len_neg]
                train_len = sampled_data1['user'].input_id.size()[0]
                batch_index = torch.arange(train_len)
                batch_index = batch_index.to(device, non_blocking=True)
                x_dict1 = multi_task_model(sampled_data1 ,stage = 'stage2')
                #print('x_dict1',x_dict1['user'])
                x_dict2 = multi_task_model(sampled_data2, stage = 'stage2')
                pos,tot,ssl_temp = ssl_block(x_dict1['user'], x_dict2['user'],batch_index,batch_index)
                #print('res',pos/tot)
                batch += 1
                ssl_temp_s_avg += ssl_temp
                infonce_loss = -torch.sum(torch.log(pos/tot))
                infonce_loss = ssl_reg * infonce_loss
                infonce_loss.backward()
                optimizer_ssl.step()
                total_loss += float(infonce_loss) 
                ssl_s_loss += float(infonce_loss) 
            ssl_temp_s_avg = ssl_temp_s_avg/batch
            print('ssl_temp_s_avg',ssl_temp_s_avg)    
            print('ssl for item seller finished')

        print(f"Epoch: {epoch:03d}, Loss: {total_loss :.4f},transe_Loss: {transe_loss :.4f},\
                BPR_u2i_Loss: {bpr_u2i_loss :.4f}\
                BPR_u2s_Loss: {bpr_u2s_loss :.4f}, SSl_s_Loss: {ssl_s_loss :.4f}, \
                SSl_i_Loss: {ssl_i_loss :.4f},SSl_is_Loss: {ssl_is_loss :.4f}")
        '''
        print(f"Epoch: {epoch:03d}, Loss: {total_loss :.4f},transe_Loss: {transe_loss :.4f},\
                BPR_u2i_Loss: {bpr_u2i_loss :.4f}\
                BPR_u2s_Loss: {bpr_u2s_loss :.4f}")
        '''
        return multi_task_model
    
    
def eval_item(test_data,multi_task_model):
    #new ver
    val_user = val_data['user'].node_id
    preds = []
    ground_truths = []
    auc = []
    n = 0
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks))}
    start = time.time()
    #num_user = val_user.size()[0]
    num_user = 100
    for u in val_user[5000:]:
        if n == 100:
            break
        pred_user = []
        gf_user = []
        all_test_id = []
        user_index = torch.where(torch.isin(val_data['user'].node_id,u))[0]
        #all_pos = val_data["user", "u2i", "item"].edge_label_index[1][val_data["user", "u2i", "item"].edge_label==1]
        all_pos_val = val_data["user", "u2i", "item"].edge_index[1][val_data["user", "u2i", "item"].edge_index[0]==user_index]
        #print('all_pos_item',all_pos_item)
        #all_pos_item = all_pos_item[torch.where(torch.isin(all_pos_item,all_pos))[0]]
        #print('all_pos_item',all_pos_item)
        #if all_pos_item.size()[0] == 0 :
        #   num_user -= 1
        #   continue
       # all_pos_id = val_data['item'].node_id[all_pos_item]
       # print('all_pos_item',all_pos_item.size())
        except_item = torch.unique(data["user", "u2i", "item"].edge_index[1][data["user", "u2i", "item"].edge_index[0]==user_index])
        print('except_item',len(except_item))
        all_pos_item = except_item[torch.where(torch.isin(except_item,all_pos_val,invert = True))[0]]
        print('all_pos_item',len(all_pos_item))
        if len(all_pos_item) == 0 :
            num_user -= 1
            n += 1
            continue
        all_pos_id = data['item'].node_id[all_pos_item]
        all_exp_item = torch.unique(torch.cat((all_pos_item,except_item)))
        #print('all_exp_item',all_exp_item.size())
        all_index = torch.arange(val_data['item'].node_id.size()[0])
        all_neg_item = torch.where(torch.isin(all_index,all_exp_item,invert = True))[0]
        print('all_neg_item',all_neg_item.size())
        all_test_item = torch.cat((all_pos_item,all_neg_item))
        test_user = user_index.expand_as(all_test_item)
        edge_label_index_u2i = torch.stack([test_user,all_test_item],dim = 0)
        pos_label = torch.tensor([1])
        pos_label = pos_label.expand_as(all_pos_item)
        neg_label = torch.tensor([0])
        neg_label = neg_label.expand_as(all_neg_item)
        edge_label_u2i = torch.cat((pos_label,neg_label))
        #print('edge_label_index_u2i',edge_label_index_u2i.size())
        val_loader_u2i = LinkNeighborLoader(
        data=val_data.edge_type_subgraph([('user', 'u2i', 'item'),('item', 'i2u', 'user')]),
        num_neighbors=[15,15],
        neg_sampling_ratio=0, 
        edge_label_index=(("user", "u2i", "item"), edge_label_index_u2i),
        edge_label=edge_label_u2i,
        batch_size=16*128,
        shuffle=False,)
        #print('edge_label_u2i',edge_label_u2i[:50])
        for sampled_data in tqdm(val_loader_u2i):
            with torch.no_grad():
                sampled_data = sampled_data.to('cpu')
                val_loader_i2s = NeighborLoader(
                    val_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[15,15],
                    batch_size=sampled_data['item'].node_id.size()[0],
                    input_nodes=('item', sampled_data['item'].node_id))
                for i, sampled_i2s in enumerate(val_loader_i2s):
                    batch_len = sampled_i2s['item'].input_id.size()[0]
                    edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                    sampled_i2s = sampled_i2s.to(device)
                    x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1',edge_attr = edge_attr)
                sampled_data['item'].x = x_dict1['item'][:batch_len]
                sampled_data = sampled_data.to(device)
                batch_pred = multi_task_model(sampled_data)
                #batch_pred = torch.sigmoid(batch_pred)
                #print('pred',batch_pred)
                pred_user.append(batch_pred)
                gf_user.append(sampled_data["user", "u2i", "item"].edge_label)
                all_test_id.append(sampled_data["item"].node_id[sampled_data["user", "u2i", "item"].edge_label_index[1]])
        pred_user = torch.cat(pred_user, dim=0).cpu().numpy()
        print('pred_user',pred_user[:100])
        gf_user = torch.cat(gf_user, dim=0).cpu().numpy()
        print('gf_user',gf_user[:100])
        all_test_id = torch.cat(all_test_id,dim = 0).cpu().numpy()
        print('all_test_id',len(all_test_id))
        r = ranklist_by_sorted(all_pos_id,all_test_id,pred_user, Ks)
        #print('r',r)
        auc.append(roc_auc_score(gf_user, pred_user))
        res = get_performance(all_pos_id, r, Ks)
        #print('res',res)
        result['precision'] += res['precision']
        result['recall'] += res['recall']
        result['ndcg'] += res['ndcg']
        result['hit_ratio'] += res['hit_ratio']
        n+=1

    print()
    result['precision'] = result['precision']/num_user
    result['recall'] = result['recall']/num_user
    result['ndcg'] = result['ndcg']/num_user
    result['hit_ratio'] = result['hit_ratio']/num_user
    print(f"Validation AUC: {np.mean(auc):.4f}")
    final_perf = "Val res=recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                     ('\t'.join(['%.5f' % r for r in result['recall']]),
                      '\t'.join(['%.5f' % r for r in result['precision']]),
                      '\t'.join(['%.5f' % r for r in result['hit_ratio']]),
                      '\t'.join(['%.5f' % r for r in result['ndcg']]))
    print(final_perf)
    print(time.time()-start)
    
def eval_seller(test_data,multi_task_model):
    #new ver
    val_user = val_data['user'].node_id
    preds = []
    ground_truths = []
    auc = []
    n = 0
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks))}
    start = time.time()
    #num_user = val_user.size()[0]
    num_user = 300
    for u in val_user[25000:]:
        if n == 300:
            break
        pred_user = []
        gf_user = []
        all_test_id = []
        user_index = torch.where(torch.isin(val_data['user'].node_id,u))[0]
        #all_pos = val_data["user", "u2i", "item"].edge_label_index[1][val_data["user", "u2i", "item"].edge_label==1]
        all_pos_val = val_data["user", "u2s", "seller"].edge_index[1][val_data["user", "u2s", "seller"].edge_index[0]==user_index]
        #print('all_pos_item',all_pos_item)
        #all_pos_item = all_pos_item[torch.where(torch.isin(all_pos_item,all_pos))[0]]
        #print('all_pos_item',all_pos_item)
        #if all_pos_item.size()[0] == 0 :
        #   num_user -= 1
        #   continue
       # all_pos_id = val_data['item'].node_id[all_pos_item]
       # print('all_pos_item',all_pos_item.size())
        except_seller = torch.unique(data["user", "u2s", "seller"].edge_index[1][data["user", "u2s", "seller"].edge_index[0]==user_index])
        print('except_seller',len(except_seller))
        all_pos_seller = except_seller[torch.where(torch.isin(except_seller,all_pos_val,invert = True))[0]]
        print('all_pos_seller',len(all_pos_seller))
        if len(all_pos_seller) == 0 :
            num_user -= 1
            n += 1
            continue
        all_pos_id = data['seller'].node_id[all_pos_seller]
        all_exp_seller = torch.unique(torch.cat((all_pos_seller,except_seller)))
        #print('all_exp_item',all_exp_item.size())
        all_index = torch.arange(val_data['seller'].node_id.size()[0])
        all_neg_seller = torch.where(torch.isin(all_index,all_exp_seller,invert = True))[0]
        print('all_neg_sller',all_neg_seller.size())
        all_test_seller = torch.cat((all_pos_seller,all_neg_seller))
        test_user = user_index.expand_as(all_test_seller)
        edge_label_index_u2s = torch.stack([test_user,all_test_seller],dim = 0)
        pos_label = torch.tensor([1])
        pos_label = pos_label.expand_as(all_pos_seller)
        neg_label = torch.tensor([0])
        neg_label = neg_label.expand_as(all_neg_seller)
        edge_label_u2s = torch.cat((pos_label,neg_label))
        #print('edge_label_index_u2i',edge_label_index_u2i.size())
        val_loader_u2s = LinkNeighborLoader(
        data=val_data.edge_type_subgraph([('user', 'u2s', 'seller'),('seller', 's2u', 'user')]),
        num_neighbors=[15,15],
        neg_sampling_ratio=0, 
        edge_label_index=(("user", "u2s", "seller"), edge_label_index_u2s),
        edge_label=edge_label_u2s,
        batch_size=16*128,
        shuffle=False,)
        #print('edge_label_u2i',edge_label_u2i[:50])
        for sampled_data in tqdm(val_loader_u2s):
            with torch.no_grad():
                sampled_data = sampled_data.to('cpu')
                val_loader_i2s = NeighborLoader(
                    val_data.edge_type_subgraph([("item", "i2s", "seller"), ("seller", "s2i", "item")]),
                    num_neighbors=[15,15],
                    batch_size=sampled_data['seller'].node_id.size()[0],
                    input_nodes=('seller', sampled_data['seller'].node_id))
                for i, sampled_i2s in enumerate(val_loader_i2s):
                    batch_len = sampled_i2s['seller'].input_id.size()[0]
                    edge_attr = torch.tensor([2,5]).to(device, non_blocking=True)
                    sampled_i2s = sampled_i2s.to(device)
                    x_dict1 = multi_task_model(sampled_i2s ,stage = 'stage1',edge_attr = edge_attr)
                sampled_data['seller'].x = x_dict1['seller'][:batch_len]
                sampled_data = sampled_data.to(device)
                batch_pred = multi_task_model(sampled_data)
                #batch_pred = torch.sigmoid(batch_pred)
                #print('pred',batch_pred)
                pred_user.append(batch_pred)
                gf_user.append(sampled_data["user", "u2s", "seller"].edge_label)
                all_test_id.append(sampled_data["seller"].node_id[sampled_data["user", "u2s", "seller"].edge_label_index[1]])
        pred_user = torch.cat(pred_user, dim=0).cpu().numpy()
        print('pred_user',pred_user[:100])
        gf_user = torch.cat(gf_user, dim=0).cpu().numpy()
        print('gf_user',gf_user[:100])
        all_test_id = torch.cat(all_test_id,dim = 0).cpu().numpy()
        print('all_test_id',len(all_test_id))
        r = ranklist_by_sorted(all_pos_id,all_test_id,pred_user, Ks)
        #print('r',r)
        auc.append(roc_auc_score(gf_user, pred_user))
        res = get_performance(all_pos_id, r, Ks)
        #print('res',res)
        result['precision'] += res['precision']
        result['recall'] += res['recall']
        result['ndcg'] += res['ndcg']
        result['hit_ratio'] += res['hit_ratio']
        n+=1

    print()
    result['precision'] = result['precision']/num_user
    result['recall'] = result['recall']/num_user
    result['ndcg'] = result['ndcg']/num_user
    result['hit_ratio'] = result['hit_ratio']/num_user
    print(f"Validation AUC: {np.mean(auc):.4f}")
    final_perf = "Val res=recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                     ('\t'.join(['%.5f' % r for r in result['recall']]),
                      '\t'.join(['%.5f' % r for r in result['precision']]),
                      '\t'.join(['%.5f' % r for r in result['hit_ratio']]),
                      '\t'.join(['%.5f' % r for r in result['ndcg']]))
    print(final_perf)
    print(time.time()-start)
    
    
def main():
    args = get_args()
    prepare(args)
    multi_task_model, ssl_block, optimizer, optimizer_ssl,device = prepare_model(args)
    data, train_data, val_data, test_data = Dataset(args.data_file,args,data_name)
    train(args,multi_task_model,train_data,data,ssl_block, optimizer, optimizer_ssl,device)
    eval_item(val_data,multi_task_model)
    eval_item(val_data,multi_task_model)
    
    
if __name__ == "__main__":
    main()
