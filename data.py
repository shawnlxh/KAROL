import pandas as pd
import torch
from torch import Tensor
from collections import defaultdict
import heapq
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def Dataset(source_path,dataset_name):
    num_row = 1000000
    df = pd.read_csv('./raw_data_food.csv', sep = '\t', nrows = num_row)
    df.head(20)
    user_filter = defaultdict(set)
    for index, row in df.iterrows():
        user_id = row['cust_id']
        item_id = row['item_id']
        if user_id not in user_filter.keys():
            user_filter[user_id] = [item_id]
        else:
            user_filter[user_id].append(item_id)
    filter_num = 5
    filtered_user_id = []
    for k, v in user_filter.items():
        if len(v) >= 5:
            filtered_user_id.append(k)
    print(filtered_user_id[:5])
    df = df.loc[df['cust_id'].isin(filtered_user_id)]
    # Create a mapping from unique user indices to range [0, num_user_nodes):
    unique_user_id = df['cust_id'].unique()
    unique_user_id = pd.DataFrame(data={
        'user_id': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })
    print("Mapping of user IDs to consecutive values:")
    print("==========================================")
    print(unique_user_id.head())
    print()
    # Create a mapping from unique item indices to range [0, num_item_nodes):
    unique_item_id = df['item_id'].unique()
    unique_item_id = pd.DataFrame(data={
        'item_id': unique_item_id,
        'mappedID': pd.RangeIndex(len(unique_item_id)),
    })
    print("Mapping of item IDs to consecutive values:")
    print("===========================================")
    print(unique_item_id.head())
    print()

    # Create a mapping from unique seller indices to range [0, num_seller_nodes):
    unique_seller_id = df['rec_seller_id'].unique()
    unique_seller_id = pd.DataFrame(data={
        'seller_id': unique_seller_id,
        'mappedID': pd.RangeIndex(len(unique_seller_id)),
    })
    print("Mapping of seller IDs to consecutive values:")
    print("==========================================")
    print(unique_seller_id.head())

    # Perform merge to obtain the edges from users and items:
    mapping_user_id = pd.merge(df['cust_id'], unique_user_id,
                                left_on='cust_id', right_on='user_id', how='left')
    mapping_user_id = torch.from_numpy(mapping_user_id['mappedID'].values)
    mapping_item_id = pd.merge(df['item_id'], unique_item_id,
                                left_on='item_id', right_on='item_id', how='left')
    mapping_item_id = torch.from_numpy(mapping_item_id['mappedID'].values)
    mapping_seller_id = pd.merge(df['rec_seller_id'], unique_seller_id,
                                left_on='rec_seller_id', right_on='seller_id', how='left')
    mapping_seller_id = torch.from_numpy(mapping_seller_id['mappedID'].values)

    # With this, we are ready to construct our `edge_index` in COO format
    # following PyG semantics:
    # relation u2i : user - item
    # relation u2s : user - seller
    # relation i2u : item - user
    # relation i2s : item - seller
    # relation s2u : seller - user
    # relation s2i : seller - item
    edge_index_user_to_item = torch.stack([mapping_user_id, mapping_item_id], dim=0)
    edge_index_user_to_seller = torch.stack([mapping_user_id, mapping_seller_id], dim=0)
    edge_index_item_to_user = torch.stack([mapping_item_id, mapping_user_id], dim=0)
    edge_index_item_to_seller = torch.stack([mapping_item_id, mapping_seller_id], dim=0)
    edge_index_seller_to_user = torch.stack([mapping_seller_id, mapping_user_id], dim=0)
    edge_index_seller_to_item = torch.stack([mapping_seller_id, mapping_item_id], dim=0)


    print("Final edge indices pointing of users, items and sellers:")
    print("=================================================")
    print(edge_index_user_to_item)
    print(edge_index_user_to_seller)
    print(edge_index_item_to_user)
    print(edge_index_item_to_seller)
    print(edge_index_seller_to_user)
    print(edge_index_seller_to_item)
    
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    df_seller_raw = spark.read.parquet('gs://p13n-storage2/user/s0v035u/mp/data/seller_features_vectorized/*')
    df_seller_pq = df_seller_raw.filter((df_seller_raw.rec_offer_type == '3P')).select('org_seller_id', 'repr')
    df_seller_pq.printSchema()
    df_seller = df_seller_pq.toPandas().drop_duplicates().rename(columns={'org_seller_id':'seller_id'})
    df_seller = df_seller.set_index('seller_id').join(unique_seller_id.set_index('seller_id'), on = 'seller_id')
    df_seller = df_seller.dropna()
    df_seller.head(3)
    data = HeteroData()

    # Save node indices:
    data["user"].node_id = torch.arange(len(unique_user_id))
    data["item"].node_id = torch.arange(len(unique_item_id))
    data["seller"].node_id = torch.arange(len(unique_seller_id))

    data["user", "u2i", "item"].edge_index = edge_index_user_to_item
    data["user", "u2s", "seller"].edge_index = edge_index_user_to_seller
    data["item", "i2u", "user"].edge_index = edge_index_item_to_user
    data["item", "i2s", "seller"].edge_index = edge_index_item_to_seller
    data["seller", "s2u", "user"].edge_index = edge_index_seller_to_user
    data["seller", "s2i", "item"].edge_index = edge_index_seller_to_item
    
    if dataset_name == 'clothes':
        # Add the node features and edge indices:
        df_item = df[['item_id','item_type']]
        item_category = df['item_type'].str.get_dummies()
        df_item_category = pd.concat([df_item, item_category], axis=1).\
            drop_duplicates(subset = ['item_id']).drop(['item_id','item_type'], axis=1)
        print(df_item_category.head(3))
        item_feat = torch.from_numpy(df_item_category.values).to(torch.float)
        print(item_feat.size())
        item_feat = normalize(item_feat)
        data["item"].x = item_feat
        item_feat_dim = item_feat.size()[1]

        df_seller = df_seller.sort_values(by=['mappedID'])
        seller_feat = []
        for spark_row in df_seller['repr'].values:
            seller_feat.append(list(spark_row.asDict().values()))
        seller_feat = np.array(seller_feat)
        seller_feat[seller_feat == None] = 0
        seller_feat = seller_feat.astype(np.float32)
        seller_feat_all = np.zeros((len(unique_seller_id), len(seller_feat[0])))
        all_included_mappedID = df_seller['mappedID'].values
        all_included_mappedID = [int(x) for x in all_included_mappedID]
        for i, mappedID in enumerate(all_included_mappedID):
            seller_feat_all[all_included_mappedID] = seller_feat[i]

        print(seller_feat_all.shape)
        seller_feat = torch.from_numpy(seller_feat_all).to(torch.float)
        seller_feat_dim = seller_feat.size()[1]
        seller_feat = normalize(seller_feat)
        data["seller"].x = seller_feat
        print(seller_feat.size())
    
    transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=0, 
    add_negative_train_samples= False,
    edge_types=[("user", "u2i", "item"), ("user", "u2s", "seller")],
    rev_edge_types=[("item", "i2u", "user"), ("seller", "s2u", "user")],)
    #data = data.edge_type_subgraph([('user', 'u2i', 'item'),('item', 'i2u', 'user')])
    train_data, val_data, test_data = transform(data)
    print("Training data:")
    print("==============")
    print(train_data)
    #print(train_data["user", "u2i", "item"].num_edges)
    print()
    print("Validation data:")
    print("================")
    print(val_data)
    print("test_data data:")
    print("================")
    print(test_data)
    
    return data, train_data, val_data, test_data