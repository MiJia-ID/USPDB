import numpy
import numpy as np
import torch
import os
import pickle

from feature.create_edge_181 import create_dis_matrix

th = 17

root_dir = '/home/mijia/EGPDI/data/'
save_files_dir = '/home/mijia/EGPDI/data/dataset_dir_181/'

test_path= root_dir + 'DNA_Test_181.txt'
dis_path= root_dir + '/dataset_dir_181/PDNA_psepos_SC.pkl'

adj_type = 'adj_SC_17_predicted'

query_ids = []

with open(test_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        if query_id[-1].islower():
            #query_id += query_id[-1]
            query_id[-1] = query_id[-1].upper()
            print(query_id, '-' * 1000)
        query_ids.append(query_id)

distance_matrixs=create_dis_matrix(dis_path,query_ids)
print(len(distance_matrixs))

# save adj_matrix of DNA_181_Test
def create_adj_pkl(query_ids,dis_matrix):
    for i in range(len(query_ids)):
        # create adj from distance matrix then creating edge_index
        binary_matrix = (dis_matrix[i] <= th).astype(int)
        sym_matrix = np.triu(binary_matrix) + np.triu(binary_matrix, 1).T
        dis_matrix[i] = sym_matrix
        adj_matrix = torch.from_numpy(dis_matrix[i])
        adj_matrix = adj_matrix.float()

        fpath = save_files_dir +'Test_181/'
        if not os.path.exists(fpath + adj_type):
            os.makedirs(fpath + adj_type)

        if not os.path.exists(fpath + adj_type + '/{}.pkl'.format(query_ids[i])):
            print(fpath + adj_type + '/{}.pkl'.format(query_ids[i]))
            with open(fpath + adj_type + '/{}.pkl'.format(query_ids[i]) , 'wb') as f:
                pickle.dump(adj_matrix, f)


create_adj_pkl(query_ids,distance_matrixs)