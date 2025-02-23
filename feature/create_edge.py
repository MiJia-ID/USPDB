# create edge_features of protein graphs on DNA_573_Train and DNA_129_Test

import pickle
import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
import os

th = 17
#这个函数用于为一组蛋白质创建距离矩阵的列表
def create_dis_matrix(dis_path,Query_ids):
    dis_load=open(dis_path,'rb')
    dis_residue=pickle.load(dis_load)
    #dis_residue['4ne1_p'] = dis_residue.pop('4ne1_pp')

    distance_matrixs=[]

    for i in Query_ids:
        residues=dis_residue[i]
        num_node = len(residues)
        residues_array = np.array(residues)
        distance_matrix = np.zeros((num_node, num_node))
        distances = np.linalg.norm(residues_array[:, np.newaxis, :] - residues_array[np.newaxis, :, :], axis=-1)
        distance_matrix[np.triu_indices(num_node, k=1)] = distances[np.triu_indices(num_node, k=1)]
        distance_matrix += distance_matrix.T        # undirected graphs, so the distance matrix is symmetrical
        distance_matrixs.append(distance_matrix)

    return distance_matrixs

def parse_residue_coordinates(pdb_file_path):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('protein', pdb_file_path)
    model = structure[0]  # 选择第一个模型
    residues = []

    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != ' ':  # 跳过非标准残基
                continue
            # 获取 CA 原子的坐标
            if 'CA' in residue:
                residues.append(residue['CA'].get_coord())

    return np.array(residues)

def create_dis_matrix_by_pdb(pdb_file_path):
    # 从 PDB 文件中解析残基坐标
    residues_array = parse_residue_coordinates(pdb_file_path)

    num_node = len(residues_array)
    distance_matrix = np.zeros((num_node, num_node))

    # 计算残基之间的欧氏距离
    distances = np.linalg.norm(residues_array[:, np.newaxis, :] - residues_array[np.newaxis, :, :], axis=-1)
    distance_matrix[np.triu_indices(num_node, k=1)] = distances[np.triu_indices(num_node, k=1)]
    distance_matrix += distance_matrix.T  # 无向图，所以距离矩阵是对称的

    return distance_matrix

def cal_edges(dis_matrix,protein_idx,th):
    
    dis_matrix_copy = dis_matrix.copy()
    binary_matrix = (dis_matrix_copy[protein_idx] <= th).astype(int)
    symmetric_matrix = np.triu(binary_matrix) + np.triu(binary_matrix, 1).T
    dis_matrix_copy[protein_idx] = symmetric_matrix
    binary_matrix = torch.from_numpy(dis_matrix_copy[protein_idx])

    mask = (binary_matrix ==1)
    radius_index_list = np.where(mask)
    radius_index_list = [list(nodes) for nodes in zip(radius_index_list[0], radius_index_list[1])]

    return radius_index_list

# calculate edge_features: distance_attrs and cos_similarity_attrs
def calculate_edge_attributes(edge_index_list, distance_matrixs, protein_idx):
    pdist = nn.PairwiseDistance(p=2, keepdim=True)
    cossim = nn.CosineSimilarity(dim=0)

    num_edges = len(edge_index_list)
    distance_attrs = torch.zeros(num_edges)
    cos_similarity_attrs = torch.zeros(num_edges)

    for i in range(num_edges):
        src_idx, dst_idx = edge_index_list[i]

        distance_matrix_src = torch.tensor(distance_matrixs[protein_idx][src_idx][0])
        distance_matrix_dst = torch.tensor(distance_matrixs[protein_idx][dst_idx][0])
        distance = pdist(distance_matrix_src, distance_matrix_dst).item()
        distance_attrs[i] = distance / 17

        distance_matrix_src_array = torch.tensor(distance_matrixs[protein_idx][src_idx])
        distance_matrix_dst_array = torch.tensor(distance_matrixs[protein_idx][dst_idx])
        cos_similarity = cossim(distance_matrix_src_array, distance_matrix_dst_array).item()
        cos_similarity_attrs[i] = (cos_similarity + 1) / 2

    return distance_attrs, cos_similarity_attrs

def get_edge_attr_train(pro_id,th,distance_matrixs):
    edge_index_list = cal_edges(distance_matrixs[:573], protein_idx=pro_id, th=th)
    distance_attrs, cos_similarity_attrs = calculate_edge_attributes(edge_index_list, distance_matrixs[:573], protein_idx=pro_id)
    edge_attr_train = torch.stack((distance_attrs, cos_similarity_attrs), dim=1)
    return edge_attr_train

def get_edge_attr_test(pro_id,th,distance_matrixs):
    edge_index_list = cal_edges(distance_matrixs[573:], protein_idx=pro_id, th=th)
    distance_attrs, cos_similarity_attrs = calculate_edge_attributes(edge_index_list, distance_matrixs[573:], protein_idx=pro_id)
    edge_attr_test = torch.stack((distance_attrs, cos_similarity_attrs), dim=1)

    return edge_attr_test





# #计算edge的时候放出来，train的时候注释掉
# root_dir = '/home/mijia/EGPDI/data/'
# train_path = root_dir + 'DNA-573_Train.txt'
# test_path = root_dir + 'DNA_Test_181.txt'
# pdb_folder_path = '/home/mijia/EGPDI/data/pdb_dir/'
# #换数据集记得换这里的地址
# dis_path= root_dir + 'dataset_dir_181/PDNA_psepos_SC.pkl'
# query_ids = []
# #读取所有测试集的id
# with open(test_path, 'r') as f:
#     text = f.readlines()
#     for i in range(0, len(text), 3):
#         query_id = text[i].strip()[1:]
#         # if query_id[-1].islower():
#         #     query_id += query_id[-1]
#         query_ids.append(query_id)
# distance_matrixs=create_dis_matrix(dis_path,query_ids)
# efeats = []
# for pro_id in range(len(query_ids)):
#     egde_feats = get_edge_attr_train(pro_id,th,distance_matrixs)
#     efeats.append(egde_feats)
#     print("正在计算")
# save_edgefeats_path = '/home/mijia/EGPDI/data/dataset_dir_181/Test_181/edge_features/EdgeFeats_predicted_SC_17_181.pkl'
# if not os.path.exists(save_edgefeats_path):
#     os.makedirs("/home/mijia/EGPDI/data/dataset_dir_181/Test_181/edge_features")
#     with open(save_edgefeats_path, 'wb') as f:
#         pickle.dump(efeats, f)
#         print("已保存至：" + save_edgefeats_path)