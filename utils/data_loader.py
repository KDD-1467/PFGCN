from urllib.request import CacheFTPHandler
import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
# train_user_set = defaultdict(list)
# test_user_set = defaultdict(list)

# def read_cf(file_name):
#     inter_mat = list()
#     lines = open(file_name, "r").readlines()
#     for l in lines:
#         tmps = l.strip()
#         inters = [int(i) for i in tmps.split(" ")]

#         u_id, pos_ids = inters[0], inters[1:]
#         pos_ids = list(set(pos_ids))
#         for i_id in pos_ids:
#             inter_mat.append([u_id, i_id])

#     return np.array(inter_mat)

# def remap_item(train_data, test_data):
#     global n_users, n_items
#     n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
#     n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

#     for u_id, i_id in train_data:
#         train_user_set[int(u_id)].append(int(i_id))
#     for u_id, i_id in test_data:
#         test_user_set[int(u_id)].append(int(i_id))

def read_triplets(file_name, file_name_valid,file_name_test): # done
    global n_entities, n_relations, n_nodes, n_users, n_items

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)
    test_triplets = np.loadtxt(file_name_test, dtype=np.int32)
    test_triplets = np.unique(test_triplets,axis=0)
    valid_triplets = np.loadtxt(file_name_valid, dtype=np.int32)
    valid_triplets = np.unique(valid_triplets,axis=0)

    n_users = max(max(can_triplets_np[:,0]), max(test_triplets[:,0]), max(valid_triplets[:,0])) + 1
    n_items = max(max(can_triplets_np[:,2]), max(test_triplets[:,2]), max(valid_triplets[:,2])) + 1
    can_triplets_np[:,2] = can_triplets_np[:,2] + n_users

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        # inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        inv_triplets_np[:,1] = can_triplets_np[:,1]
        # consider two additional relations --- 'interact' and 'be interacted'
        # can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        # inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        # can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()


    n_entities = n_users + n_items
    n_nodes = n_entities
    n_relations = max(max(triplets[:, 1]),max(test_triplets[:, 1]), max(valid_triplets[:,1])) + 1

    # n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    # n_nodes = n_entities + n_users
    
    return triplets

def build_graph(triplets): # done
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)
    # triplets = triplets[np.where(triplets[:,-1] == 1)]
    # triplets = triplets[:,-1]
    # print("Begin to load interaction triples ...")
    # for u_id, i_id in tqdm(train_data, ascii=True):
    #     rd[0].append([u_id, i_id])

    print("\nBegin to load user-text-item triples ...")
    for h_id, r_id, t_id, label in tqdm(triplets, ascii=True):
        if label == 1:
            ckg_graph.add_edge(h_id, t_id, key=r_id)
            rd[r_id].append([h_id, t_id])

    return ckg_graph, rd



def build_sparse_relational_graph(relation_dict): # done
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        # if r_id == 0:
        #     cf = np_mat.copy()
        #     cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
        #     vals = [1.] * len(cf)
        #     adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        # else:
        cf = np_mat.copy()
        cf[:,1] = cf[:,1] + n_users # [0, n_items) -> [n_users, n_users+n_items)
        vals = [1.] * len(cf)
        adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    # norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    # mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list

def load_data(model_args):
    global args
    args = model_args
    # directory = args.data_path + args.dataset + '/'

    # print('reading train and test user-item set ...')
    # train_cf = read_cf(directory + 'train.txt')
    # test_cf = read_cf(directory + 'test.txt')
    # remap_item(train_cf, test_cf)

    print('read user-text-item tripltes...')
    if args.dataset == 'data_a':
        triplets = read_triplets('./data_a/triplets_train_new.txt', './data_a/triplets_valid_new.txt','./data_a/triplets_test_new.txt')
    elif args.dataset == 'data_b':
        triplets = read_triplets('./data_b/triplets_train_new.txt', './data_b/triplets_valid_new.txt', './data_b/triplets_test_new.txt')
    else :
        raise NotImplementedError
    print('building the graph ...')
    graph, relation_dict = build_graph(triplets)
    
    # print('building the adj mat ...')
    # adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }

    return n_params, graph
        #    [adj_mat_list, norm_mat_list, mean_mat_list]